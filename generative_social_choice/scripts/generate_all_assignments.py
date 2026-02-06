#!/usr/bin/env python3
"""
Generate assignments for all existing runs under data/results/statements/.

Discovers run directories and, for each run that has a utility matrix,
runs compute_assignments (default: slate_size=5, generated-only, no seed).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from generative_social_choice.utils.helper_functions import get_base_dir_path, get_results_paths
from generative_social_choice.scripts.compute_assignments import run as compute_assignments_run
from generative_social_choice.scripts.compute_assignments import VOTING_ALGORITHMS


def discover_runs(
    statements_dir: Path,
    embedding_types: tuple[str, ...] = ("llm", "fish"),
) -> list[tuple[str, str]]:
    """Return list of (run_id, embedding_type) for runs that have a utility matrix."""
    pairs: list[tuple[str, str]] = []
    for run_dir in sorted(statements_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        # Skip subdirs that are not run_ids (e.g. stray generated_with_* at top level)
        if run_id.startswith("generated_with_"):
            continue
        for emb in embedding_types:
            paths = get_results_paths(
                labelling_model="4o-mini",
                embedding_type=emb,
                baseline=False,
                base_dir=statements_dir.parent,
                run_id=run_id,
            )
            if paths["utility_matrix_file"].exists():
                pairs.append((run_id, emb))
                break
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slate_size",
        type=int,
        default=5,
        help="Slate size (default: 5).",
    )
    parser.add_argument(
        "--include_seed",
        action="store_true",
        help="Include the 6 seed statements in the candidate set (saves with _with-seed).",
    )
    parser.add_argument(
        "--embedding_types",
        nargs="+",
        default=["llm", "fish"],
        help="Embedding types to consider when discovering runs (default: llm fish).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print which runs would be processed.",
    )
    args = parser.parse_args()

    base = get_base_dir_path()
    statements_dir = base / "data" / "results" / "statements"
    if not statements_dir.exists():
        raise SystemExit(f"Statements directory not found: {statements_dir}")

    runs = discover_runs(statements_dir, tuple(args.embedding_types))
    if not runs:
        print("No runs with utility matrices found.")
        return

    print(f"Found {len(runs)} run(s): {runs}")
    if args.dry_run:
        return

    for run_id, embedding_type in runs:
        print(f"\n{'='*60}\nRun id={run_id!r} embedding_type={embedding_type!r}\n{'='*60}")
        result_paths = get_results_paths(
            labelling_model="4o-mini",
            embedding_type=embedding_type,
            baseline=False,
            run_id=run_id,
        )
        for name, algo in VOTING_ALGORITHMS.items():
            print(f"  {algo.name} ...")
            compute_assignments_run(
                slate_size=args.slate_size,
                voting_algotirhm=algo,
                utility_matrix_file=result_paths["utility_matrix_file"],
                statement_id_file=result_paths["statement_id_file"],
                assignment_file=result_paths["assignments"] / f"{name}.json",
                include_seed=args.include_seed,
                verbose=False,
            )
    print("\nDone.")


if __name__ == "__main__":
    main()
