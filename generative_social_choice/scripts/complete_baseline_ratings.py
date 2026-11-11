"""
Collect the ratings missing from the ballot profiles of the baseline runs (paper_replication/generate_slate.py).

The baseline pipeline only rates each round's candidates by the voters still unassigned in that round,
so the utility matrix in each run's info.csv is incomplete. This script queries the missing
(voter, candidate) pairs with the same discriminative query and stores them next to info.csv
(see `COMPLETION_RATINGS_FILENAME`), without modifying any of the run's outputs.
Use `get_completed_baseline_utility_matrices` to load the completed profiles.

The script can be interrupted and rerun: ratings which are already stored are not queried again.
"""
import ast
import json
import argparse
import random
import threading
import concurrent.futures
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from generative_social_choice.ratings.rating_generation import get_agents, Rating
from generative_social_choice.ratings.utility_matrix import (
    get_baseline_run_dirs,
    read_baseline_info_csv,
    COMPLETION_RATINGS_FILENAME,
    COMPLETION_LOGS_FILENAME,
)
from generative_social_choice.slates.slate_generation import LOGS_COLUMNS
from generative_social_choice.queries.query_chatbot_personalization import ChatbotPersonalizationAgent


# Discriminative model used by the baseline runs (see generate_slate.py)
MODEL = "gpt-4o-mini-2024-07-18"

COMPLETION_LOGS_COLUMNS = list(dict.fromkeys(LOGS_COLUMNS)) + ["query_type", "agent_id", "statement"]


def check_statements(run_dir: Path, info: pd.DataFrame):
    """Check that the statements recovered from info.csv match the slate written to committee.txt"""
    with open(run_dir / "committee.txt", "r", encoding="utf-8") as f:
        slate = ast.literal_eval(f.read().split("\n")[1])
    chosen = list(info["round_chosen"].dropna().sort_values().index)
    if chosen != slate:
        raise ValueError(f"Statements in info.csv don't match the slate in committee.txt of {run_dir}")


def read_completed_ratings(run_dir: Path) -> list[Rating]:
    ratings_file = run_dir / COMPLETION_RATINGS_FILENAME
    if not ratings_file.exists():
        return []
    with open(ratings_file, "r", encoding="utf-8") as f:
        return [Rating(**json.loads(line)) for line in f]


def get_missing_pairs(run_dir: Path, info: pd.DataFrame) -> list[tuple[str, str]]:
    """Get the (agent_id, statement) pairs which are neither rated in info.csv nor in the completion ratings"""
    completed = {(rating.agent_id, rating.statement) for rating in read_completed_ratings(run_dir)}
    approvals = info[[col for col in info.columns if col.startswith("approval_")]]
    missing = []
    for statement, row in approvals.iterrows():
        for col in row.index[row.isna()]:
            agent_id = col.removeprefix("approval_")
            if (agent_id, statement) not in completed:
                missing.append((agent_id, statement))
    return missing


def query_pairs(
        agents: dict[str, ChatbotPersonalizationAgent],
        pairs: list[tuple[str, str]],
        num_threads: int,
        on_result=None,
    ) -> list[Rating]:
    """Rate the given (agent_id, statement) pairs, calling on_result(rating, logs) for each one as it completes"""
    # Ordering by agent lets consecutive queries share the agent's prompt prefix for prompt caching
    pairs = sorted(pairs)
    ratings = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_pair = {
            executor.submit(agents[agent_id].get_approval, statement): (agent_id, statement)
            for agent_id, statement in pairs
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_pair), total=len(future_to_pair)):
            approval, logs = future.result()
            agent_id, statement = future_to_pair[future]
            rating = Rating(agent_id=agent_id, statement=statement, approval=approval)
            ratings.append(rating)
            if on_result is not None:
                on_result(rating, logs)
    return ratings


def complete_run(run_dir: Path, agents: dict[str, ChatbotPersonalizationAgent], num_threads: int):
    info = read_baseline_info_csv(run_dir)
    check_statements(run_dir, info)
    missing = get_missing_pairs(run_dir, info)
    print(f"{run_dir.name}: querying {len(missing)} missing ratings")

    ratings_file = run_dir / COMPLETION_RATINGS_FILENAME
    logs_file = run_dir / COMPLETION_LOGS_FILENAME
    lock = threading.Lock()

    def save(rating: Rating, logs: list[dict]):
        # Write each rating as soon as it arrives so that an interrupted run loses nothing
        with lock:
            with open(ratings_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rating.to_dict()) + "\n")
            logs = [{**log, "agent_id": rating.agent_id, "statement": rating.statement} for log in logs]
            pd.DataFrame(logs, columns=COMPLETION_LOGS_COLUMNS).to_csv(
                logs_file, mode="a", header=not logs_file.exists(), index=False
            )

    query_pairs(agents, missing, num_threads=num_threads, on_result=save)


def print_missing(run_dir: Path):
    info = read_baseline_info_csv(run_dir)
    check_statements(run_dir, info)
    missing = get_missing_pairs(run_dir, info)
    missing_per_statement = pd.Series([statement for _, statement in missing]).value_counts()
    per_round = (
        info.assign(missing=missing_per_statement.reindex(info.index).fillna(0).astype(int))
        .groupby("round_added")["missing"]
        .agg(lambda counts: sorted(counts.tolist()))
    )
    print(f"{run_dir.name}: {len(missing)} missing; per candidate, by round added: {per_round.to_dict()}")


def calibrate(run_dir: Path, agents: dict[str, ChatbotPersonalizationAgent], num_pairs: int, num_threads: int, seed: int = 0):
    """Re-query ratings which are already in info.csv and compare (nothing is written to disk)"""
    info = read_baseline_info_csv(run_dir)
    check_statements(run_dir, info)
    approvals = info[[col for col in info.columns if col.startswith("approval_")]]
    known = {
        (col.removeprefix("approval_"), statement): approval
        for statement, row in approvals.iterrows()
        for col, approval in row.dropna().items()
    }
    pairs = random.Random(seed).sample(sorted(known), num_pairs)
    ratings = query_pairs(agents, pairs, num_threads=num_threads)
    diffs = pd.Series([rating.approval - known[(rating.agent_id, rating.statement)] for rating in ratings])
    print(
        f"{run_dir.name}: re-queried {len(diffs)} known ratings; "
        f"mean diff (new - old) {diffs.mean():+.4f} (SE {diffs.sem():.4f}), "
        f"mean abs diff {diffs.abs().mean():.4f}, max abs diff {diffs.abs().max():.4f}, "
        f"exact matches {(diffs.abs() < 1e-9).sum()}/{len(diffs)}"
    )
    return diffs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_ids",
        type=int,
        nargs="+",
        default=None,
        help="Zero-indexed baseline runs to complete (see get_baseline_run_dirs). If not provided, all runs are completed.",
    )

    parser.add_argument(
        "--embedding_type",
        type=str,
        default="llm",
        choices=["llm", "seed_statement"],
        help="Which set of baseline runs to complete (see get_baseline_run_dirs).",
    )

    parser.add_argument(
        "--candidate_pool",
        type=str,
        default="round",
        choices=["round", "persistent"],
        help="Which set of baseline runs to complete (see get_baseline_run_dirs). The persistent-pool runs were "
             "rated with LF line endings in the prompts and the earlier round-only runs with CRLF, so the "
             "missing ratings are queried with the same line endings.",
    )

    parser.add_argument(
        "--num_threads",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print the number of missing ratings per run.",
    )

    parser.add_argument(
        "--calibrate",
        type=int,
        default=None,
        help="Instead of completing the runs, re-query this many known ratings per run and compare them to info.csv. Nothing is written.",
    )

    args = parser.parse_args()

    run_dirs = get_baseline_run_dirs(args.run_ids, args.embedding_type, args.candidate_pool)

    if args.dry_run:
        for run_dir in run_dirs.values():
            print_missing(run_dir)
    else:
        # Same prompt line endings as the runs' own ratings (see --candidate_pool)
        agents = {
            agent.get_id(): agent
            for agent in get_agents(model=MODEL, lf_line_endings=args.candidate_pool == "persistent")
        }
        for run_dir in run_dirs.values():
            if args.calibrate is not None:
                calibrate(run_dir, agents, num_pairs=args.calibrate, num_threads=args.num_threads)
            else:
                complete_run(run_dir, agents, num_threads=args.num_threads)
