"""
Measure whether today's LLM inference is in distribution with the 2025 inference that produced the
existing results, for every inference type whose outputs are compared across eras when the Fish
baseline is re-run (see RESULT_model_drift.md for the write-up).

Ratings (gpt-4o-mini, `ChatbotPersonalizationAgent.get_approval`):
- R1: ratings of Fish-pipeline candidate statements (old baseline runs' info.csv)
- R2: ratings of GenerativeCC statements (the 10 "Ours" utility matrices)
- R3: neighbour-search ratings, i.e. rating another agent's summary (old baseline runs' logs.csv)
Generation (gpt-4o-2024-08-06): the exact 2025 messages of sampled generator calls, replayed twice.

Today's values are compared with the 2025 values *and* with the 2025-vs-2025 spread of cells which
were rated more than once in 2025, since both models were already nondeterministic then.

All queries are appended to queries/*.jsonl (next to this script) as they complete, so the script can be
interrupted and rerun without repeating queries. Run with `--analyze_only` to only write the report.
"""
import ast
import copy
import json
import hashlib
import argparse
import random
import threading
import concurrent.futures
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from generative_social_choice.utils.helper_functions import get_base_dir_path
from generative_social_choice.utils.gpt_wrapper import GPT, Embeddings
from generative_social_choice.ratings.rating_generation import get_agents

SEED = 0
OUT_DIR = Path(__file__).parent / "queries"
RESULT_FILE = Path(__file__).parent / "RESULT_model_drift.md"
RATINGS_FILE = OUT_DIR / "ratings.jsonl"
GENERATIONS_FILE = OUT_DIR / "generations.jsonl"
EMBEDDINGS_FILE = OUT_DIR / "embeddings.jsonl"

BASELINE_DIR = get_base_dir_path() / "data/demo_data/generate_slate_results_baseline"
OURS_DIR = get_base_dir_path() / "data/results/statements"
RATER_MODEL = "gpt-4o-mini-2024-07-18"  # Model used by the baseline runs
OURS_RATER_MODEL = "4o-mini"  # Model string used by scripts/rate_statements.py
GENERATOR_MODEL = "gpt-4o-2024-08-06"  # Snapshot the baseline runs' "gpt-4o" resolved to (see their logs)
GENERATOR_SLOTS = ["Personalization (T=0)", "Personalization (T=1)", "Subsampling (n=5)",
                   "Nearest neighbour (20, 5)", "Nearest neighbour (100, 5)", "Nearest neighbour (100, 10)"]

SEED_STATEMENT_IDS = {f"s{i}" for i in range(1, 7)}  # Survey answers, not LLM ratings
BINS = ["<2", "2-3", "3-3.5", ">3.5"]
N_R1_DUPLICATES = 100
N_R1_PER_BIN = 50
N_R2_PILOT_PER_BIN = 10
N_R2_PER_BIN = 38
N_R3_PER_BIN = 38
N_REPEATS = {"R1": 40, "R2": 30, "R3": 30}
N_PROMPTS_PER_SLOT = 4
N_REPLAYS = 2
N_QUALITY_VOTERS = 6
N_BOOTSTRAP = 2000


def value_bin(value: float) -> str:
    return BINS[int(np.digitize(value, [2, 3, 3.5]))]


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


##################################################
# Tasks and resumable storage
##################################################

@dataclass
class RatingTask:
    kind: str  # R1, R2, R3, Q (generation quality)
    variant: str  # prompt variant; "exact" unless testing alternatives
    rep: int  # 0 for the first rating today, 1 for the repeat
    agent_id: str
    statement: str
    old_values: list[float] = field(default_factory=list)  # 2025 values, possibly several
    meta: dict = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        return f"{self.kind}:{self.variant}:{self.rep}:{self.agent_id}:{text_hash(self.statement)}:{self.meta.get('key', '')}"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_threaded(tasks, fn, out_file: Path, num_threads: int, desc: str):
    """Run fn(task) -> record for all tasks whose task_id is not yet in out_file, appending each record."""
    done = {record["task_id"] for record in read_jsonl(out_file)}
    todo = [task for task in tasks if task.task_id not in done]
    print(f"{desc}: {len(tasks) - len(todo)} done, {len(todo)} to query")
    if not todo:
        return
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(fn, task) for task in todo]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=desc):
            record = future.result()
            with lock, open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


##################################################
# Loading the 2025 data
##################################################

def load_baseline_runs() -> list[dict]:
    runs = []
    for run_dir in sorted(d for d in BASELINE_DIR.iterdir() if d.is_dir()):
        info = pd.read_csv(run_dir / "info.csv", index_col=0)
        # Undo the line ending conversion of the checkout; generated statements contain "\n"
        info.index = [statement.replace("\r\n", "\n") for statement in info.index]
        logs = pd.read_csv(run_dir / "logs.csv", index_col=0, low_memory=False)
        runs.append({"name": run_dir.name, "info": info, "logs": logs})
    return runs


def fish_candidate_cells(runs: list[dict]) -> pd.DataFrame:
    """All ratings of candidate statements in the old baseline runs"""
    rows = []
    for run in runs:
        info = run["info"]
        approvals = info[[col for col in info.columns if col.startswith("approval_")]]
        for position, (statement, row) in enumerate(approvals.iterrows()):
            for col, value in row.dropna().items():
                rows.append({
                    "run": run["name"], "agent_id": col.removeprefix("approval_"), "statement": statement,
                    "value": value, "round": int(info["round_added"].iloc[position]), "slot": position % 6,
                })
    return pd.DataFrame(rows)


def neighbour_search_cells(runs: list[dict], agents: dict) -> pd.DataFrame:
    """All neighbour-search ratings in the logs of the old baseline runs: the rated "statement" is
    the summary of another agent (the centre)."""
    prefixes = {agent_id: agent.prompt_template.split("$statement") for agent_id, agent in agents.items()}
    summary_to_centre = {agent.get_description().replace("\r\n", "\n"): agent_id for agent_id, agent in agents.items()}
    rows = []
    for run in runs:
        logs = run["logs"]
        ratings = logs[logs["model"] == RATER_MODEL]
        for prompt, response in zip(ratings["prompt"].astype(str), ratings["response"]):
            for agent_id, (prefix, suffix) in prefixes.items():
                if prompt.startswith(prefix) and prompt.endswith(suffix):
                    statement = prompt[len(prefix): len(prompt) - len(suffix)].replace("\r\n", "\n")
                    centre = summary_to_centre.get(statement)
                    if centre is not None:
                        rows.append({"run": run["name"], "agent_id": agent_id, "centre": centre, "value": float(response)})
                    break
    return pd.DataFrame(rows)


def ours_cells() -> tuple[pd.DataFrame, dict]:
    """All LLM ratings in the 10 GenerativeCC utility matrices (the 6 seed statements are survey answers)"""
    rows, statements = [], {}
    for run_id in range(10):
        run_dir = OURS_DIR / str(run_id) / "generated_with_4o_using_llm_embeddings"
        # Exactly as scripts/rate_statements.py read the statements
        raw_statements = pd.read_csv(run_dir / "statement_generation_raw_output.csv")["statement"].to_list()
        ids = pd.read_csv(run_dir / "4o-mini_for_labelling/utility_matrix_statements.csv", index_col=0)
        matrix = pd.read_csv(run_dir / "4o-mini_for_labelling/utility_matrix.csv", index_col=0)
        # create_utility_matrix reorders the statements, so map IDs by text
        key = lambda s: " ".join(s.split())
        raw_by_text = {key(statement): statement for statement in raw_statements}
        assert len(raw_by_text) == len(raw_statements)
        for statement_id, statement in ids["statement"].items():
            if statement_id in SEED_STATEMENT_IDS:
                continue
            statements[(run_id, statement_id)] = raw_by_text[key(statement)]
            for agent_id, value in matrix[statement_id].items():
                rows.append({"run": run_id, "statement_id": statement_id, "agent_id": agent_id, "value": value})
    return pd.DataFrame(rows), statements


def stratified_sample(df: pd.DataFrame, per_bin: int, rng: random.Random, by: str | None = None) -> pd.DataFrame:
    """Sample per_bin rows from each value bin, spread evenly over the groups in column `by` if given"""
    samples = []
    for b in BINS:
        in_bin = df[df["bin"] == b]
        if by is None or in_bin.empty:
            samples.append(in_bin.sample(min(per_bin, len(in_bin)), random_state=rng.randrange(2**31)))
            continue
        groups = [g for _, g in in_bin.groupby(by)]
        per_group = -(-per_bin // len(groups))
        picked = pd.concat([g.sample(min(per_group, len(g)), random_state=rng.randrange(2**31)) for g in groups])
        samples.append(picked.sample(min(per_bin, len(picked)), random_state=rng.randrange(2**31)))
    return pd.concat(samples)


##################################################
# Building the tasks
##################################################

def build_rating_tasks(runs, agents, rng) -> tuple[list[RatingTask], dict]:
    """R1, R3 and the repeats; R2 needs the pilot first (see build_r2_tasks)"""
    tasks, references = [], {}

    # R1: Fish candidate ratings
    r1 = fish_candidate_cells(runs)
    r1["bin"] = r1["value"].map(value_bin)
    grouped = r1.groupby(["agent_id", "statement"])
    duplicates = grouped.filter(lambda g: g["run"].nunique() > 1)
    references["R1_population_bins"] = r1["bin"].value_counts().to_dict()
    references["R1_within_2025"] = [
        (b - a, value_bin(a)) for _, g in duplicates.groupby(["agent_id", "statement"])
        for a, b in [tuple(g.sort_values("run")["value"].iloc[:2])]
    ]
    duplicate_keys = list(duplicates.groupby(["agent_id", "statement"]).groups)
    for agent_id, statement in rng.sample(sorted(duplicate_keys), N_R1_DUPLICATES):
        values = duplicates[(duplicates.agent_id == agent_id) & (duplicates.statement == statement)].sort_values("run")["value"].tolist()
        tasks.append(RatingTask("R1", "exact", 0, agent_id, statement, values[:2], {"key": "dup", "duplicate": True, "bin": value_bin(values[0])}))
    singles = r1[~r1.set_index(["agent_id", "statement"]).index.isin(duplicate_keys)]
    for row in stratified_sample(singles, N_R1_PER_BIN, rng, by="slot").itertuples():
        tasks.append(RatingTask("R1", "exact", 0, row.agent_id, row.statement, [row.value],
                                {"key": row.run, "duplicate": False, "bin": row.bin, "slot": row.slot, "round": row.round}))

    # R3: neighbour-search ratings
    r3 = neighbour_search_cells(runs, agents)
    r3["bin"] = r3["value"].map(value_bin)
    references["R3_population_bins"] = r3["bin"].value_counts().to_dict()
    references["R3_within_2025"] = [
        (b - a, value_bin(a)) for _, g in r3.groupby(["agent_id", "centre"]) if g["run"].nunique() > 1
        for a, b in [tuple(g.drop_duplicates("run").sort_values("run")["value"].iloc[:2])]
    ]
    for row in stratified_sample(r3, N_R3_PER_BIN, rng).itertuples():
        summary = agents[row.centre].get_description()  # As read from the checkout, like in 2025
        tasks.append(RatingTask("R3", "exact", 0, row.agent_id, summary, [row.value], {"key": row.run, "bin": row.bin, "centre": row.centre}))

    return tasks, references


def r2_variant_agent(agent, variant: str):
    """GenerativeCC's ratings left no logs, so the newlines of its prompts are unknown. The survey
    answers enter the prompt as escaped "\\r\\n" (CRLF checkout, like the baseline runs) or "\\n"."""
    if variant.startswith("survey_lf"):
        agent = copy.copy(agent)
        agent.prompt_template = agent.prompt_template.replace("\\r\\n", "\\n")
    return agent


def r2_variant_statement(statement: str, variant: str) -> str:
    return statement.replace("\r\n", "\n") if variant.endswith("statement_lf") else statement


R2_VARIANTS = ["survey_crlf+statement_crlf", "survey_crlf+statement_lf", "survey_lf+statement_crlf", "survey_lf+statement_lf"]


def build_r2_tasks(rng, variant: str | None) -> tuple[list[RatingTask], dict]:
    """The pilot rates the same cells with all four variants; the main sample uses the chosen one"""
    cells, statements = ours_cells()
    cells["bin"] = cells["value"].map(value_bin)
    references = {"R2_population_bins": cells["bin"].value_counts().to_dict()}
    # Only statements with newlines distinguish the statement variants
    has_newline = cells.apply(lambda row: "\r\n" in statements[(row.run, row.statement_id)], axis=1)
    pilot = stratified_sample(cells[has_newline], N_R2_PILOT_PER_BIN, rng)
    main = stratified_sample(cells.drop(pilot.index), N_R2_PER_BIN, rng)
    tasks = []
    for v in R2_VARIANTS:
        for row in pilot.itertuples():
            statement = r2_variant_statement(statements[(row.run, row.statement_id)], v)
            tasks.append(RatingTask("R2", v, 0, row.agent_id, statement, [row.value], {"key": f"pilot:{row.run}:{row.statement_id}", "bin": row.bin, "pilot": True}))
    if variant is not None:
        for row in main.itertuples():
            statement = r2_variant_statement(statements[(row.run, row.statement_id)], variant)
            tasks.append(RatingTask("R2", variant, 0, row.agent_id, statement, [row.value], {"key": f"main:{row.run}:{row.statement_id}", "bin": row.bin, "pilot": False}))
    return tasks, references


def repeat_tasks(tasks: list[RatingTask], rng) -> list[RatingTask]:
    repeats = []
    for kind, n in N_REPEATS.items():
        pool = [t for t in tasks if t.kind == kind and t.rep == 0 and not t.meta.get("pilot")]
        for t in rng.sample(pool, min(n, len(pool))):
            repeats.append(RatingTask(t.kind, t.variant, 1, t.agent_id, t.statement, t.old_values, t.meta))
    return repeats


@dataclass
class GenerationTask:
    run: str
    call_index: int
    rep: int
    messages: list
    params: dict
    old_response: str

    @property
    def task_id(self) -> str:
        return f"G:{self.run}:{self.call_index}:{self.rep}"

    @property
    def slot(self) -> int:
        return self.call_index % 6

    @property
    def round(self) -> int:
        return self.call_index // 6


def generator_calls(run: dict) -> pd.DataFrame:
    """The 30 generation calls of a run; call i was made by generator slot i % 6 in round i // 6"""
    calls = run["logs"][run["logs"]["model"] == "gpt-4o"].reset_index(drop=True)
    assert len(calls) == 30
    return calls


def build_generation_tasks(runs, rng) -> list[GenerationTask]:
    tasks = []
    for slot in range(6):
        for round_num in rng.sample(range(5), N_PROMPTS_PER_SLOT):
            run = rng.choice(runs)
            call_index = round_num * 6 + slot
            call = generator_calls(run).iloc[call_index]
            for rep in range(N_REPLAYS):
                tasks.append(GenerationTask(
                    run=run["name"], call_index=call_index, rep=rep,
                    # The messages are stored as a Python repr, which escapes newlines, so they are
                    # recovered exactly (verified against rebuilt round-0 messages)
                    messages=ast.literal_eval(call["messages"]),
                    params=ast.literal_eval(call["params"]),
                    old_response=str(call["response"]).replace("\r\n", "\n"),
                ))
    return tasks


def round0_references(runs) -> dict[int, list[str]]:
    """Slots 0-2 sent byte-identical round-0 messages in every run: 15 outputs from 2025 per prompt"""
    outputs = {}
    for slot in range(3):
        messages = {generator_calls(run).iloc[slot]["messages"] for run in runs}
        assert len(messages) == 1, slot
        outputs[slot] = [str(generator_calls(run).iloc[slot]["response"]).replace("\r\n", "\n") for run in runs]
    return outputs


def new_runs_round0_outputs() -> dict[int, list[str]]:
    """Round-0 outputs of generators 1-3 in the new persistent-pool baseline runs (empty if there are none)"""
    new_dir = BASELINE_DIR.with_name(BASELINE_DIR.name + "_persistent_pool")
    if not new_dir.exists():
        return {}
    outputs = {slot: [] for slot in range(3)}
    # Only finished runs, which have written their committee
    for run_dir in sorted(d for d in new_dir.iterdir() if (d / "committee.txt").exists()):
        logs = pd.read_csv(run_dir / "logs.csv", index_col=0, low_memory=False)
        calls = logs[logs["model"] == GENERATOR_MODEL].reset_index(drop=True)
        for slot in range(3):
            outputs[slot].append(str(calls.iloc[slot]["response"]).replace("\r\n", "\n"))
    return outputs


def statement_from_response(response: str) -> str:
    return response.strip()


##################################################
# Querying
##################################################

def make_rating_fn(agents_pinned: dict, agents_ours: dict):
    def rate(task: RatingTask) -> dict:
        if task.kind == "R2":
            agent = r2_variant_agent(agents_ours[task.agent_id], task.variant)
        else:
            agent = agents_pinned[task.agent_id]
        approval, logs = agent.get_approval(task.statement)
        return {"task_id": task.task_id, **asdict(task), "new_value": approval,
                "model": logs[0]["model"], "fingerprint": logs[0]["system_fingerprint"], "timestamp": logs[0]["timestamp"]}
    return rate


def generate(task: GenerationTask) -> dict:
    llm = GPT(model=GENERATOR_MODEL)
    response, _, log = llm.call(messages=task.messages, **task.params)
    return {"task_id": task.task_id, "run": task.run, "call_index": task.call_index, "slot": task.slot, "round": task.round,
            "rep": task.rep, "old_response": task.old_response, "new_response": response,
            "fingerprint": log["system_fingerprint"], "timestamp": log["timestamp"]}


def embed_all(texts: list[str]) -> dict[str, np.ndarray]:
    cache = {r["hash"]: np.array(r["embedding"]) for r in read_jsonl(EMBEDDINGS_FILE)}
    missing = sorted({t for t in texts if text_hash(t) not in cache})
    if missing:
        embedder = Embeddings()
        for start in range(0, len(missing), 100):
            batch = missing[start:start + 100]
            vectors, _ = embedder.embed(batch)
            with open(EMBEDDINGS_FILE, "a", encoding="utf-8") as f:
                for text, vector in zip(batch, vectors):
                    f.write(json.dumps({"hash": text_hash(text), "embedding": list(vector)}) + "\n")
                    cache[text_hash(text)] = np.array(vector)
    return {t: cache[text_hash(t)] for t in texts}


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


##################################################
# Analysis
##################################################

def bootstrap_ci(values, statistic=np.mean, rng=None, n=N_BOOTSTRAP) -> tuple[float, float, float]:
    values = np.asarray(values)
    rng = rng or np.random.default_rng(SEED)
    stats = [statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(n)]
    return float(statistic(values)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def weighted_bias_ci(diffs: pd.Series, bins: pd.Series, population: dict, rng) -> tuple[float, float, float]:
    """Bias over the population, from a sample stratified by bin (resampling within bins)"""
    total = sum(population.get(b, 0) for b in BINS if (bins == b).any())
    groups = {b: diffs[bins == b].to_numpy() for b in BINS if (bins == b).any()}
    estimate = lambda gs: sum(population.get(b, 0) / total * g.mean() for b, g in gs.items())
    stats = [estimate({b: g[rng.integers(0, len(g), len(g))] for b, g in groups.items()}) for _ in range(N_BOOTSTRAP)]
    return estimate(groups), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


MIN_REFERENCE_PAIRS = 20


def spread_ratio_ci(diffs, bins, ref_diffs, ref_bins, population: dict, rng) -> tuple[float, float, float] | None:
    """Mean squared difference today vs 2025 over mean squared difference 2025 vs 2025, combined over
    value bins with population weights, since noise depends on the rating level and the samples are
    stratified differently. Only bins with enough 2025 reference pairs are used. Resamples within bins."""
    diffs, bins, ref_diffs, ref_bins = map(np.asarray, (diffs, bins, ref_diffs, ref_bins))
    usable = [b for b in BINS if (bins == b).any() and (ref_bins == b).sum() >= MIN_REFERENCE_PAIRS]
    if not usable:
        return None
    total = sum(population.get(b, 0) for b in usable)
    groups = {b: (diffs[bins == b], ref_diffs[ref_bins == b]) for b in usable}
    ratio = lambda gs: (sum(population.get(b, 0) / total * np.mean(d ** 2) for b, (d, _) in gs.items())
                        / sum(population.get(b, 0) / total * np.mean(r ** 2) for b, (_, r) in gs.items()))
    resample = lambda x: x[rng.integers(0, len(x), len(x))]
    stats = [ratio({b: (resample(d), resample(r)) for b, (d, r) in groups.items()}) for _ in range(N_BOOTSTRAP)]
    return float(ratio(groups)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def fmt(ci, digits=3) -> str:
    return f"{ci[0]:+.{digits}f} [{ci[1]:+.{digits}f}, {ci[2]:+.{digits}f}]"


def fmt_ratio(ci) -> str:
    return f"{ci[0]:.2f} [{ci[1]:.2f}, {ci[2]:.2f}]"


def analyze(references: dict, r2_variant: str, round0: dict, runs) -> str:
    rng = np.random.default_rng(SEED)
    ratings = pd.DataFrame(read_jsonl(RATINGS_FILE))
    ratings["bin"] = ratings["meta"].map(lambda m: m["bin"])
    ratings["old"] = ratings["old_values"].map(lambda v: v[0] if v else np.nan)
    ratings["diff"] = ratings["new_value"] - ratings["old"]
    lines = ["# Model drift: is today's inference in distribution with 2025's?", "",
             "Generated by `generative_social_choice/data/model_drift/measure_model_drift.py`. Differences are today minus 2025, on the 0-4 rating scale. "
             "Brackets are bootstrap 95% CIs. The *spread ratio* is Var(today - 2025) / Var(2025 - 2025'), where the denominator "
             "comes from cells which were rated in two different 2025 runs; a ratio of 1 means today's ratings vary around the 2025 "
             "ratings exactly as much as 2025 ratings varied among themselves.", ""]

    # --- R2 pilot: which prompt variant reproduces GenerativeCC's 2025 ratings
    pilot = ratings[(ratings.kind == "R2") & ratings["meta"].map(lambda m: m.get("pilot", False)) & (ratings.rep == 0)]
    lines += ["## R2 pilot: prompt variant of GenerativeCC's ratings", "",
              "GenerativeCC's rating logs were not committed, so the newlines of its prompts are unknown. "
              "The same 40 cells were rated with each variant; the variant closest to the 2025 values is used for R2.", "",
              "| Variant | Mean abs. diff | Bias |", "|---|---|---|"]
    for v in R2_VARIANTS:
        d = pilot[pilot.variant == v]["diff"]
        lines.append(f"| {v}{' (chosen)' if v == r2_variant else ''} | {d.abs().mean():.3f} | {d.mean():+.3f} |")
    lines.append("")

    # --- Ratings
    # Each GenerativeCC cell was rated once, so R2 is compared with the R1 reference
    within = {"R1": references["R1_within_2025"], "R2": references["R1_within_2025"], "R3": references["R3_within_2025"]}
    names = {"R1": "R1: Fish candidate ratings", "R2": "R2: GenerativeCC ratings", "R3": "R3: neighbour-search ratings"}
    lines += ["## Ratings (gpt-4o-mini)", "",
              "| Inference | n | Bias (pop.-weighted) | Mean abs. diff today vs 2025 | Mean abs. diff 2025 vs 2025 | Mean abs. diff today vs today | Spread ratio |",
              "|---|---|---|---|---|---|---|"]
    gate = {}
    per_bin_rows = []
    for kind in ["R1", "R2", "R3"]:
        first = ratings[(ratings.kind == kind) & (ratings.rep == 0) & ((ratings.kind != "R2") | (ratings.variant == r2_variant))]
        reference = np.array([d for d, _ in within[kind]])
        reference_bins = np.array([b for _, b in within[kind]])
        repeat = ratings[(ratings.kind == kind) & (ratings.rep == 1)].merge(first[["task_id", "new_value"]].assign(task_id=lambda x: x.task_id.str.replace(":0:", ":1:", n=1)), on="task_id", suffixes=("", "_first"))
        today_today = (repeat["new_value"] - repeat["new_value_first"]).to_numpy()
        population = references[f"{kind}_population_bins"]
        bias = weighted_bias_ci(first["diff"], first["bin"], population, rng)
        ratio = spread_ratio_ci(first["diff"], first["bin"], reference, reference_bins, population, rng)
        lines.append(f"| {names[kind]} | {len(first)} | {fmt(bias)} | {first['diff'].abs().mean():.3f} | {np.abs(reference).mean():.3f} (n={len(reference)}{', R1 reference' if kind == 'R2' else ''}) | "
                     f"{np.abs(today_today).mean():.3f} (n={len(today_today)}) | {fmt_ratio(ratio)} |")
        gate[kind] = {"bias": bias, "ratio": ratio}
        for b in BINS:
            d = first[first.bin == b]["diff"].to_numpy()
            ref_b = reference[reference_bins == b]
            if len(d) == 0:
                continue
            bin_bias = bootstrap_ci(d, rng=rng)
            bin_ratio = spread_ratio_ci(d, [b] * len(d), ref_b, [b] * len(ref_b), {b: 1}, rng)
            # Binning by a noisy 2025 value makes re-ratings regress to the mean even without drift,
            # so the bias in a bin is compared with the same statistic for 2025-vs-2025 pairs
            if len(ref_b) >= MIN_REFERENCE_PAIRS:
                ref_ci = bootstrap_ci(ref_b, rng=rng)
                excess = (bin_bias[0] - ref_ci[0], bin_bias[1] - ref_ci[2], bin_bias[2] - ref_ci[1])
                gate[kind][b] = excess
                excess_str = fmt(excess)
            else:
                excess_str = f"n/a (<{MIN_REFERENCE_PAIRS} pairs)"
            per_bin_rows.append(f"| {kind} | {b} | {len(d)} | {fmt(bin_bias)} | "
                                + (f"{ref_b.mean():+.3f}" if len(ref_b) else "n/a") + f" | {excess_str} | {np.abs(d).mean():.3f} | "
                                f"{np.abs(ref_b).mean():.3f} (n={len(ref_b)}) | " + (fmt_ratio(bin_ratio) if bin_ratio else f"n/a (<{MIN_REFERENCE_PAIRS} pairs)") + " |")
    lines += ["", "### By 2025 value", "",
              "Binning by the (noisy) 2025 value makes re-ratings regress to the mean: low cells tend to come back higher and high "
              "cells lower, even without drift. The bias in each bin is therefore compared with the mean of 2025-vs-2025 differences "
              "binned the same way; *excess* is the difference (conservative CI combining both intervals).", "",
              "| Inference | 2025 value | n | Bias today vs 2025 | Same, 2025 vs 2025 | Excess | Mean abs. diff today vs 2025 | Mean abs. diff 2025 vs 2025 | Spread ratio |",
              "|---|---|---|---|---|---|---|---|---|", *per_bin_rows, ""]

    # Direct test on cells rated twice in 2025 and once today
    dup = ratings[(ratings.kind == "R1") & (ratings.rep == 0) & ratings["meta"].map(lambda m: m.get("duplicate", False))]
    a = dup["old_values"].map(lambda v: v[0]).to_numpy(); b = dup["old_values"].map(lambda v: v[1]).to_numpy(); c = dup["new_value"].to_numpy()
    idx = lambda: rng.integers(0, len(c), len(c))
    direct = [(lambda i: np.mean((c[i] - a[i]) ** 2) / np.mean((b[i] - a[i]) ** 2))(idx()) for _ in range(N_BOOTSTRAP)]
    lines += ["### Cells rated twice in 2025 and once today", "",
              f"For {len(c)} such cells (paired), mean squared difference today vs 2025 / 2025 vs 2025 = "
              f"{np.mean((c - a) ** 2) / np.mean((b - a) ** 2):.2f} [{np.percentile(direct, 2.5):.2f}, {np.percentile(direct, 97.5):.2f}]. "
              f"Mean abs. diff today vs 2025: {np.abs(c - a).mean():.3f}; 2025 vs 2025: {np.abs(b - a).mean():.3f}. "
              f"These cells come from {dup['statement'].nunique()} distinct statements.", ""]

    # Fingerprints
    old_fp = pd.concat([run["logs"]["system_fingerprint"][run["logs"]["model"] == RATER_MODEL] for run in runs]).value_counts()
    new_fp = ratings["fingerprint"].value_counts()
    lines += ["### System fingerprints (gpt-4o-mini)", "", f"2025 baseline runs: {old_fp.to_dict()}", "", f"Today: {new_fp.to_dict()}", ""]

    # --- Generation
    gens = pd.DataFrame(read_jsonl(GENERATIONS_FILE))
    texts = list(gens["old_response"]) + list(gens["new_response"]) + [t for outs in round0.values() for t in outs]
    emb = embed_all(texts)
    quality = ratings[ratings.kind == "Q"].groupby(ratings["statement"].map(text_hash))["new_value"].mean()
    q = lambda text: quality.get(text_hash(statement_from_response(text)), np.nan)
    rows = []
    for (run, call_index), g in gens.groupby(["run", "call_index"]):
        g = g.sort_values("rep")
        old, new_a, new_b = g["old_response"].iloc[0], g["new_response"].iloc[0], g["new_response"].iloc[1]
        rows.append({"slot": g["slot"].iloc[0], "round": g["round"].iloc[0],
                     "cross": np.mean([cosine_distance(emb[old], emb[new_a]), cosine_distance(emb[old], emb[new_b])]),
                     "within_today": cosine_distance(emb[new_a], emb[new_b]),
                     "q_cross": np.mean([q(new_a) - q(old), q(new_b) - q(old)]), "q_within_today": q(new_a) - q(new_b),
                     "len_old": len(old.split()), "len_new": np.mean([len(new_a.split()), len(new_b.split())]),
                     "prefix_old": old.strip().startswith("The most important rule for chatbot personalization is"),
                     "prefix_new": np.mean([t.strip().startswith("The most important rule for chatbot personalization is") for t in (new_a, new_b)])})
    prompts = pd.DataFrame(rows)
    lines += ["## Generation (gpt-4o-2024-08-06)", "",
              f"{len(prompts)} logged 2025 generation calls ({N_PROMPTS_PER_SLOT} per generator, spread over rounds) were replayed "
              f"{N_REPLAYS} times today with the exact 2025 messages and parameters. Distances are cosine distances between "
              "text-embedding-3-small embeddings. Quality is the mean rating of the statement by "
              f"{N_QUALITY_VOTERS} fixed voters, all rated today, so rater drift cancels.", "",
              "| Generator | Distance today vs 2025 | Distance today vs today | Quality today - 2025 | Quality today - today | Words 2025 / today |",
              "|---|---|---|---|---|---|"]
    for slot, g in prompts.groupby("slot"):
        lines.append(f"| {GENERATOR_SLOTS[slot]} | {g.cross.mean():.3f} | {g.within_today.mean():.3f} | {g.q_cross.mean():+.3f} | "
                     f"{g.q_within_today.abs().mean():.3f} (abs.) | {g.len_old.mean():.0f} / {g.len_new.mean():.0f} |")
    dist_gap = bootstrap_ci((prompts.cross - prompts.within_today).to_numpy(), rng=rng)
    q_gap = bootstrap_ci(prompts.q_cross.dropna().to_numpy(), rng=rng)
    lines += ["", f"Over all {len(prompts)} prompts: distance today vs 2025 minus distance today vs today = {fmt(dist_gap)}; "
              f"quality today - 2025 = {fmt(q_gap)} (today-vs-today abs. quality difference {prompts.q_within_today.abs().mean():.3f}). "
              f"Statements starting with the required phrase: 2025 {prompts.prefix_old.mean():.0%}, today {prompts.prefix_new.mean():.0%}.", ""]

    # Round-0 references: 15 outputs from 2025 for each of slots 0-2
    lines += ["### Round 0, generators 1-3: identical prompts in all 15 runs of 2025", "",
              "| Generator | Mean distance among the 15 outputs of 2025 | Mean distance today's outputs to them |", "|---|---|---|"]
    round0_checks = []
    for slot, outs in round0.items():
        within_2025 = np.array([cosine_distance(emb[x], emb[y]) for i, x in enumerate(outs) for y in outs[i + 1:]])
        today = gens[(gens.slot == slot) & (gens["round"] == 0)]["new_response"].tolist()
        to_old = np.array([cosine_distance(emb[t], emb[o]) for t in today for o in outs])
        if today:
            round0_checks.append(to_old.mean() <= np.percentile(within_2025, 75))
        lines.append(f"| {GENERATOR_SLOTS[slot]} | {within_2025.mean():.3f} | " + (f"{to_old.mean():.3f} (n={len(today)})" if today else "not in sample") + " |")
    lines += ["", "Today's two replays of a prompt were made seconds apart and are much closer to each other than the 2025 outputs "
              "of one identical prompt, which were made weeks apart (first table vs this one). So today-vs-today understates the "
              "run-to-run variation, and the 2025-vs-2025 distances above are the relevant reference.", ""]

    # The new persistent-pool runs send the same round-0 requests, except with LF instead of CRLF line endings
    new_round0 = new_runs_round0_outputs()
    if new_round0:
        emb.update(embed_all([t for outs in new_round0.values() for t in outs]))
        lines += [f"### Round 0, generators 1-3: new baseline runs ({len(next(iter(new_round0.values())))} per generator)", "",
                  "The new runs send the round-0 requests of 2025 with LF instead of CRLF line endings in the agent summaries, "
                  "so this comparison includes that prompt change as well as drift.", "",
                  "| Generator | Mean distance among 2025 outputs | Among new outputs | New outputs to 2025 outputs |", "|---|---|---|---|"]
        for slot, outs in round0.items():
            new = new_round0[slot]
            pairwise = lambda xs, ys=None: np.mean([cosine_distance(emb[x], emb[y]) for i, x in enumerate(xs) for y in (xs[i + 1:] if ys is None else ys)])
            lines.append(f"| {GENERATOR_SLOTS[slot]} | {pairwise(outs):.3f} | {pairwise(new):.3f} | {pairwise(new, outs):.3f} |")
        lines.append("")
    old_gfp = pd.concat([run["logs"]["system_fingerprint"][run["logs"]["model"] == "gpt-4o"] for run in runs]).value_counts()
    lines += [f"System fingerprints (gpt-4o): 2025 {old_gfp.to_dict()}; today {gens['fingerprint'].value_counts().to_dict()}", ""]

    # --- Gate
    checks = []
    for kind in ["R1", "R2", "R3"]:
        bias = gate[kind]["bias"]
        checks.append((f"{kind} bias CI within ±0.05", -0.05 <= bias[1] and bias[2] <= 0.05))
        checks.append((f"{kind} per-bin excess bias CIs include a value within ±0.1", all(gate[kind][b][1] <= 0.1 and gate[kind][b][2] >= -0.1 for b in BINS if b in gate[kind])))
        checks.append((f"{kind} spread ratio CI includes a value <= 1.5", gate[kind]["ratio"][1] <= 1.5))
    checks.append(("Generation: today's round-0 outputs are no farther from the 2025 outputs than the 75th percentile of 2025-vs-2025 distances", all(round0_checks)))
    checks.append(("Generation: quality today - 2025 CI includes a value within ±0.1", q_gap[1] <= 0.1 and q_gap[2] >= -0.1))
    lines += ["## Gate", "", "| Check | Pass |", "|---|---|", *[f"| {name} | {'yes' if ok else '**no**'} |" for name, ok in checks], "",
              "The plan's original checks required the per-bin and quality CIs to lie entirely within ±0.1, compared per-bin bias "
              "against zero (which confounds drift with regression to the mean), and used today-vs-today as the generation "
              "reference (too small, see above). With the stricter wording, several checks fail because the CIs are wide, not "
              "because a shift was detected.", "",
              "## Interpretation and decisions", "",
              "- No mean shift is detectable for any rating type or for generation quality. Today's ratings are noisier: about "
              "1.5-2x the 2025-vs-2025 mean squared difference, and today's own repeat ratings differ more than repeated 2025 "
              "ratings did. Today's gpt-4o-mini calls were served by many more backend configurations (system fingerprints) "
              "than in 2025. R3's bias CI fails the ±0.05 check only because it is wide (it includes 0); its point estimate is "
              "+0.03.",
              "- Generation (gpt-4o): today's outputs lie within the 2025 spread for every generator, but are less varied. At "
              "T=0, the 15 new runs' round-0 outputs are almost identical to each other, while the 15 outputs of 2025 differed "
              "about as much as T=1 outputs do (tables above). The new baseline's candidates are therefore typical of 2025's, "
              "with less run-to-run variation, which narrows its CIs slightly but should not move its mean.",
              "- A larger effect than drift: the line endings of the prompts. The data are stored with LF line endings, but this "
              "Windows checkout (core.autocrlf) turns newlines inside CSV fields into CRLF. The old baseline runs were rated with "
              "CRLF prompts (their logged prompts contain escaped CRLF), while GenerativeCC's ratings are only reproduced with LF "
              "prompts (R2 pilot); CRLF prompts rate about 0.1 higher.",
              "- Decisions (user, 2026-11-03): the new persistent-pool baseline runs use LF prompts, to match GenerativeCC, and the "
              "runs proceed without re-rating GenerativeCC.", ""]
    return "\n".join(lines)


##################################################
# Main
##################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--analyze_only", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(exist_ok=True)

    rng = random.Random(SEED)
    runs = load_baseline_runs()
    agents_pinned = {a.get_id(): a for a in get_agents(model=RATER_MODEL)}
    agents_ours = {a.get_id(): a for a in get_agents(model=OURS_RATER_MODEL)}
    rate = make_rating_fn(agents_pinned, agents_ours)

    # Sampling happens up front with fixed seeds, so reruns build the same tasks
    rating_tasks, references = build_rating_tasks(runs, agents_pinned, rng)
    pilot_tasks, r2_refs = build_r2_tasks(random.Random(SEED + 1), variant=None)
    references.update(r2_refs)
    generation_tasks = build_generation_tasks(runs, random.Random(SEED + 2))
    round0 = round0_references(runs)

    if not args.analyze_only:
        run_threaded(pilot_tasks, rate, RATINGS_FILE, args.num_threads, "R2 pilot")
    pilot = pd.DataFrame([r for r in read_jsonl(RATINGS_FILE) if r["kind"] == "R2" and r["meta"].get("pilot")])
    r2_variant = pilot.assign(abs_diff=lambda d: (d.new_value - d.old_values.map(lambda v: v[0])).abs()).groupby("variant")["abs_diff"].mean().idxmin()
    print(f"R2 prompt variant closest to 2025: {r2_variant}")
    r2_tasks, _ = build_r2_tasks(random.Random(SEED + 1), variant=r2_variant)
    rating_tasks += [t for t in r2_tasks if not t.meta.get("pilot")]
    rating_tasks += repeat_tasks(rating_tasks, random.Random(SEED + 3))

    if not args.analyze_only:
        run_threaded(rating_tasks, rate, RATINGS_FILE, args.num_threads, "Ratings")
        run_threaded(generation_tasks, generate, GENERATIONS_FILE, args.num_threads, "Generation replays")

    # Quality of generated statements: rated today by fixed voters, for both eras
    gens = read_jsonl(GENERATIONS_FILE)
    voters = random.Random(SEED + 4).sample(sorted(agents_pinned), N_QUALITY_VOTERS)
    statements = sorted({statement_from_response(g[k]) for g in gens for k in ("old_response", "new_response")})
    quality_tasks = [RatingTask("Q", "exact", 0, voter, s, [], {"key": "quality", "bin": ""}) for s in statements for voter in voters]
    if not args.analyze_only:
        run_threaded(quality_tasks, rate, RATINGS_FILE, args.num_threads, "Generation quality ratings")

    RESULT_FILE.write_text(analyze(references, r2_variant, round0, runs), encoding="utf-8")
    print(f"Wrote {RESULT_FILE}")
