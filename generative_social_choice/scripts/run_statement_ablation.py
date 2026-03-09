import argparse
import ast
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from generative_social_choice.scripts.compute_assignments import (
    VOTING_ALGORITHMS,
    run as run_assignments,
)
from generative_social_choice.utils.helper_functions import get_results_paths


def extract_generator_name(generation_method: str) -> str:
    """Extract the class name from a generation method string."""
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)", generation_method or "")
    return match.group(1) if match else generation_method


def parse_agent_ids(agent_ids_raw: object) -> frozenset[str]:
    """Parse the `agent_ids` column from statement_generation_raw_output.csv."""
    if pd.isna(agent_ids_raw) or agent_ids_raw == "":
        return frozenset()
    parsed = ast.literal_eval(str(agent_ids_raw))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list for agent_ids, got: {type(parsed)}")
    return frozenset(str(x) for x in parsed)


def normalize_statement(statement: str) -> str:
    return " ".join(str(statement).split())


def cluster_key(agent_ids: frozenset[str]) -> str:
    joined = ",".join(sorted(agent_ids))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]


@dataclass
class ExclusionRule:
    generator: str
    scope: Literal["any", "all", "cluster"] = "any"

    @classmethod
    def parse(cls, raw_rule: str) -> "ExclusionRule":
        """
        Parse rule format:
          - GeneratorName
          - GeneratorName:scope

        `scope` is one of: any, all, cluster
        """
        parts = raw_rule.split(":", 1)
        generator = parts[0].strip()
        if not generator:
            raise ValueError(f"Invalid empty generator in rule: {raw_rule!r}")
        scope = "any"
        if len(parts) >= 2 and parts[1].strip():
            scope = parts[1].strip()
        if scope not in {"any", "all", "cluster"}:
            raise ValueError(
                f"Invalid scope {scope!r} in rule {raw_rule!r}. "
                "Expected one of: any, all, cluster."
            )
        return cls(generator=generator, scope=scope)


def load_exclusion_rules(
    inline_rules: list[str], rules_json_file: Optional[Path]
) -> list[ExclusionRule]:
    parsed_rules = [ExclusionRule.parse(rule) for rule in inline_rules]
    if rules_json_file is None:
        return parsed_rules

    data = json.loads(rules_json_file.read_text())
    if not isinstance(data, list):
        raise ValueError("Rules JSON must be a list of objects.")

    for item in data:
        if not isinstance(item, dict) or "generator" not in item:
            raise ValueError(
                "Each rule in JSON must be an object with at least 'generator'."
            )
        parsed_rules.append(
            ExclusionRule(
                generator=item["generator"],
                scope=item.get("scope", "any"),
            )
        )
    for rule in parsed_rules:
        if rule.scope not in {"any", "all", "cluster"}:
            raise ValueError(
                f"Invalid scope {rule.scope!r} for generator {rule.generator!r}. "
                "Expected one of: any, all, cluster."
            )
    return parsed_rules


def build_generation_df(raw_df: pd.DataFrame, all_voters: frozenset[str]) -> pd.DataFrame:
    df = raw_df.copy()
    df["method_name"] = df["generation_method"].map(extract_generator_name)
    df["voter_ids_set"] = df["agent_ids"].map(parse_agent_ids)
    df["scope"] = df["voter_ids_set"].map(
        lambda ids: "all" if ids and ids == all_voters else "cluster"
    )
    df["cluster_key"] = df["voter_ids_set"].map(cluster_key)
    df["statement_norm"] = df["statement"].map(normalize_statement)
    return df


def rule_matches(rule: ExclusionRule, row: pd.Series) -> bool:
    generator_match = rule.generator in {row["method_name"], row["generation_method"]}
    if not generator_match:
        return False
    if rule.scope != "any" and rule.scope != row["scope"]:
        return False
    return True


def apply_llm_subsample_per_condition(
    generation_df: pd.DataFrame,
    llm_subsample_per_condition: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Keep up to `llm_subsample_per_condition` rows for each LLMGenerator condition.

    A condition is defined by the set of voters the generator was run on, represented
    by `cluster_key` (this includes the all-voters condition and each cluster).
    """
    if llm_subsample_per_condition <= 0:
        raise ValueError("llm_subsample_per_condition must be > 0.")

    df = generation_df.copy()
    df["is_excluded_subsample"] = False

    llm_mask = df["method_name"] == "LLMGenerator"
    if not llm_mask.any():
        return df

    rng = random.Random(random_seed)
    llm_ix = df[llm_mask].index.to_list()

    groups: dict[str, list[int]] = {}
    for ix in llm_ix:
        c_key = str(df.loc[ix, "cluster_key"])
        groups.setdefault(c_key, []).append(ix)

    for _, group_indices in groups.items():
        if len(group_indices) <= llm_subsample_per_condition:
            continue
        keep_ix = set(rng.sample(group_indices, llm_subsample_per_condition))
        for ix in group_indices:
            if ix not in keep_ix:
                df.loc[ix, "is_excluded_subsample"] = True

    return df


def print_available_contexts(df: pd.DataFrame) -> None:
    print("\nAvailable generator contexts:\n")
    grouped = (
        df.groupby(["method_name", "scope", "cluster_key", "voter_ids_set"])
        .size()
        .reset_index(name="num_statements")
        .sort_values(["method_name", "scope", "num_statements"], ascending=[True, True, False])
    )
    for _, row in grouped.iterrows():
        voter_ids = sorted(list(row["voter_ids_set"]))
        print(
            f"- generator={row['method_name']} | scope={row['scope']} | "
            f"cluster_key={row['cluster_key']} | voters={len(voter_ids)} | "
            f"num_statements={row['num_statements']}"
        )
        print(f"  voter_ids={voter_ids}")
    print()


def run_ablation(
    run_id: Optional[str],
    generation_model: str,
    embedding_type: str,
    labelling_model: str,
    slate_size: int,
    ignore_initial: bool,
    ablation_name: str,
    exclusion_rules: list[ExclusionRule],
    list_contexts_only: bool,
    llm_subsample_per_condition: Optional[int],
    random_seed: int,
) -> None:
    result_paths = get_results_paths(
        labelling_model=labelling_model,
        generation_model=generation_model,
        embedding_type=embedding_type,
        baseline=False,
        run_id=run_id,
    )

    raw_output_file = result_paths["statement_generation_raw_output_file"]
    source_statement_file = result_paths["statement_id_file"]
    source_utility_file = result_paths["utility_matrix_file"]

    if not raw_output_file.exists():
        raise FileNotFoundError(f"Could not find {raw_output_file}")
    if not source_statement_file.exists():
        raise FileNotFoundError(f"Could not find {source_statement_file}")
    if not source_utility_file.exists():
        raise FileNotFoundError(f"Could not find {source_utility_file}")

    raw_df = pd.read_csv(raw_output_file)
    utility_df = pd.read_csv(source_utility_file, index_col=0)
    statement_df = pd.read_csv(source_statement_file, index_col=0)
    all_voters = frozenset(str(v) for v in utility_df.index.tolist())

    generation_df = build_generation_df(raw_df=raw_df, all_voters=all_voters)

    if list_contexts_only:
        print_available_contexts(generation_df)
        return

    if not exclusion_rules and llm_subsample_per_condition is None:
        raise ValueError(
            "No filtering was configured. Provide --exclude and/or "
            "--llm_subsample_per_condition."
        )

    if llm_subsample_per_condition is not None:
        generation_df = apply_llm_subsample_per_condition(
            generation_df=generation_df,
            llm_subsample_per_condition=llm_subsample_per_condition,
            random_seed=random_seed,
        )
    else:
        generation_df["is_excluded_subsample"] = False

    generation_df["is_excluded"] = generation_df.apply(
        lambda row: any(rule_matches(rule, row) for rule in exclusion_rules),
        axis=1,
    )
    generation_df["is_excluded"] = (
        generation_df["is_excluded"] | generation_df["is_excluded_subsample"]
    )

    # Keep a statement if at least one provenance row survives.
    per_statement = generation_df.groupby("statement_norm")["is_excluded"].all()

    statement_df["statement_norm"] = statement_df["statement"].map(normalize_statement)
    statement_df["is_generated"] = statement_df["statement_norm"].isin(per_statement.index)
    statement_df["drop_statement"] = statement_df["statement_norm"].map(
        lambda key: bool(per_statement.get(key, False))
    )

    filtered_statement_df = statement_df[~statement_df["drop_statement"]].copy()
    filtered_statement_df = filtered_statement_df.drop(
        columns=["statement_norm", "is_generated", "drop_statement"]
    )

    kept_ids = filtered_statement_df.index.tolist()
    filtered_utility_df = utility_df[kept_ids].copy()

    num_candidates = len(kept_ids) - (6 if ignore_initial else 0)
    if num_candidates < slate_size:
        raise ValueError(
            f"Not enough candidates after filtering: {num_candidates} available "
            f"(slate_size={slate_size}, ignore_initial={ignore_initial})."
        )

    ablation_root = (
        result_paths["results_dir"] / "ablations" / ablation_name / f"{labelling_model}_for_labelling"
    )
    assignments_dir = ablation_root / "assignments"
    assignments_dir.mkdir(parents=True, exist_ok=True)

    filtered_statement_file = ablation_root / "utility_matrix_statements.csv"
    filtered_utility_file = ablation_root / "utility_matrix.csv"
    filtered_raw_output_file = (
        result_paths["results_dir"] / "ablations" / ablation_name / "statement_generation_raw_output.csv"
    )
    metadata_file = result_paths["results_dir"] / "ablations" / ablation_name / "ablation_metadata.json"

    filtered_statement_df.to_csv(filtered_statement_file)
    filtered_utility_df.to_csv(filtered_utility_file)
    generation_df[~generation_df["is_excluded"]][
        ["statement", "generation_method", "agent_ids"]
    ].to_csv(filtered_raw_output_file, index=False)

    summary = {
        "source": {
            "run_id": run_id,
            "generation_model": generation_model,
            "embedding_type": embedding_type,
            "labelling_model": labelling_model,
            "results_dir": str(result_paths["results_dir"]),
        },
        "ablation_name": ablation_name,
        "llm_subsample_per_condition": llm_subsample_per_condition,
        "random_seed": random_seed,
        "exclusion_rules": [
            {
                "generator": rule.generator,
                "scope": rule.scope,
            }
            for rule in exclusion_rules
        ],
        "counts": {
            "source_statements_total": int(len(statement_df)),
            "source_generated_rows": int(len(generation_df)),
            "source_unique_generated_statements": int(generation_df["statement_norm"].nunique()),
            "excluded_by_llm_subsample_rows": int(generation_df["is_excluded_subsample"].sum()),
            "excluded_generation_rows": int(generation_df["is_excluded"].sum()),
            "filtered_statements_total": int(len(filtered_statement_df)),
            "filtered_generated_rows": int((~generation_df["is_excluded"]).sum()),
        },
        "paths": {
            "utility_matrix_file": str(filtered_utility_file),
            "statement_id_file": str(filtered_statement_file),
            "assignments_dir": str(assignments_dir),
        },
    }
    metadata_file.write_text(json.dumps(summary, indent=2))

    for name, algo in VOTING_ALGORITHMS.items():
        print(f"Running assignment algorithm: {name}")
        run_assignments(
            slate_size=slate_size,
            voting_algotirhm=algo,
            utility_matrix_file=filtered_utility_file,
            statement_id_file=filtered_statement_file,
            assignment_file=assignments_dir / f"{name}.json",
            ignore_initial_statements=ignore_initial,
            verbose=False,
        )

    print("\nAblation finished.")
    print(f"- statements kept: {len(filtered_statement_df)}")
    print(f"- assignment outputs: {assignments_dir}")
    print(f"- metadata: {metadata_file}")


def parse_run_ids(run_id: Optional[str], run_ids_raw: Optional[str]) -> list[Optional[str]]:
    """Parse run ID inputs from either --run_id or --run_ids."""
    if run_id is not None and run_ids_raw is not None:
        raise ValueError("Please provide either --run_id or --run_ids, not both.")

    if run_ids_raw is not None:
        run_ids = [x.strip() for x in run_ids_raw.split(",") if x.strip()]
        if not run_ids:
            raise ValueError("--run_ids was provided but no valid run IDs were found.")
        return run_ids

    return [run_id]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run statement ablations by excluding statement generators, without rerunning "
            "statement generation or ratings."
        )
    )
    parser.add_argument("--run_id", type=str, default=None, help="Single run ID under data/results/statements.")
    parser.add_argument(
        "--run_ids",
        type=str,
        default=None,
        help="Comma-separated run IDs, e.g. '0,1,2' or 'fish_0,fish_1'.",
    )
    parser.add_argument(
        "--generation_model",
        type=str,
        choices=["4o", "4o-mini"],
        default="4o",
        help="Generation model directory tag.",
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        choices=["llm", "seed_statement", "fish"],
        default="llm",
        help="Embedding type directory tag.",
    )
    parser.add_argument(
        "--labelling_model",
        type=str,
        choices=["4o", "4o-mini"],
        default="4o-mini",
        help="Labelling model directory tag.",
    )
    parser.add_argument("--slate_size", type=int, default=5, help="Slate size for assignment algorithms.")
    parser.add_argument(
        "--ignore_initial",
        action="store_true",
        help="Ignore first 6 survey statements during slate selection.",
    )
    parser.add_argument(
        "--ablation_name",
        type=str,
        required=True,
        help="Name of this ablation; used for output directory naming.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help=(
            "Exclusion rule. Format: Generator[:scope], "
            "where scope is one of {any,all,cluster}."
        ),
    )
    parser.add_argument(
        "--exclude_rules_json",
        type=Path,
        default=None,
        help="Optional JSON file containing exclusion rules.",
    )
    parser.add_argument(
        "--list_contexts",
        action="store_true",
        help="List available generator contexts (name/scope/voter_ids) and exit.",
    )
    parser.add_argument(
        "--llm_subsample_per_condition",
        type=int,
        default=None,
        help=(
            "If provided, randomly keep only this many LLMGenerator statements per condition "
            "(all voters and each cluster separately)."
        ),
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed used for LLM per-condition subsampling.",
    )

    args = parser.parse_args()

    rules = load_exclusion_rules(inline_rules=args.exclude, rules_json_file=args.exclude_rules_json)
    run_ids = parse_run_ids(run_id=args.run_id, run_ids_raw=args.run_ids)

    for rid in run_ids:
        print(f"\n=== Running ablation for run_id={rid} ===")
        run_ablation(
            run_id=rid,
            generation_model=args.generation_model,
            embedding_type=args.embedding_type,
            labelling_model=args.labelling_model,
            slate_size=args.slate_size,
            ignore_initial=args.ignore_initial,
            ablation_name=args.ablation_name,
            exclusion_rules=rules,
            list_contexts_only=args.list_contexts,
            llm_subsample_per_condition=args.llm_subsample_per_condition,
            random_seed=args.random_seed,
        )


if __name__ == "__main__":
    main()
