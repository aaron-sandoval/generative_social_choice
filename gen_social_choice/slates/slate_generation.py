from pathlib import Path
import pandas as pd
import os
from typing import Optional, List, Tuple
from tqdm import tqdm
import concurrent.futures

from generative_social_choice.utils.helper_functions import (
    get_time_string,
    get_base_dir_path,
)
from generative_social_choice.queries.query_interface import Agent, Generator
import math

LOGS_COLUMNS = [
    "model",
    "params",
    "prompt",
    "system_prompt",
    "messages",
    "response",
    "completion",
    "timestamp",
    "system_fingerprint",
    # For now, easiest way to get logs to work is manually specify any cols potentially used
    # otherwise, IO is really slow
    "json_decode_error",
    "validation_error",
    "response",
]  # same as LLMLog


def initialize_logging(
    full_log_dir: Optional[Path], k: int, n: int
) -> tuple[Path, Path, Path, list[dict], list[dict]]:
    """Initialize logging directory. Read existing contents of logs.csv if they exist (sometimes summaries etc are prepopulated)"""
    # If log_dir already exists (e.g. has been prepopulated with some initial logs from initial_statements), use that
    # Otherwise, create new log_dir
    if full_log_dir is None:
        full_log_dir = (
            get_base_dir_path() / "experiment_logs" / f"{get_time_string()}_k={k}_n={n}"
        )
    if not Path(full_log_dir).exists():
        os.makedirs(full_log_dir)
    logs_path = full_log_dir / "logs.csv"
    info_path = full_log_dir / "info.csv"
    # If logs.csv doesn't exist create new logs.csv with appropriate cols (so we can append)
    if not logs_path.exists():
        pd.DataFrame(columns=LOGS_COLUMNS).to_csv(logs_path)
    else:
        # Set cols order to be LOGS_COLUMNS to make appending work
        logs_df = pd.read_csv(logs_path)
        pd.DataFrame(logs_df, columns=LOGS_COLUMNS).to_csv(logs_path)
    # Create info.csv with appropriate cols
    assert not info_path.exists()
    pd.DataFrame(columns=["round_added", "source"]).to_csv(info_path)
    return full_log_dir, logs_path, info_path


def generate_statements(
    generators: list[Generator],
    unmatched_agents: list[Agent],
    logs_path: Path,
    info_path: Path,
    round_num: int,
    num_threads: int,
) -> set:
    """Use generative query to generate statements."""
    candidates = set()
    logs = []
    infos = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_threads,
    ) as executor:
        future_to_data = {
            executor.submit(moderator.generate, unmatched_agents): {
                "moderator": moderator,
            }
            for moderator_idx, moderator in enumerate(generators)
        }

        for future in concurrent.futures.as_completed(future_to_data):
            query1_responses, query1_logs = future.result()
            moderator = future_to_data[future]["moderator"]
            logs.append(query1_logs)  # write to logs after all threads finish

            for i, query1_response in enumerate(query1_responses):
                source = f"{moderator.__class__.__name__}_{i}"
                candidates.add(query1_response)
                infos.append(
                    [
                        {
                            "candidate": query1_response,
                            "round_added": round_num,
                            "source": source,
                        }
                    ]
                )

    for query1_logs in logs:
        pd.DataFrame(query1_logs, columns=LOGS_COLUMNS).to_csv(
            logs_path, mode="a", header=False
        )
    for info in infos:
        pd.DataFrame(info).to_csv(info_path, mode="a", header=False, index=0)

    return candidates


def evaluate_candidates(
    unmatched_agents: List[Agent],
    candidates: set,
    utilities: dict,
    logs_path: Path,
    info_path: Path,
    num_threads: int,
    coalition_size: int,
) -> Tuple[dict, dict[str, float]]:
    """Use discriminative query to evaluate how much unmatched agents like the candidate statements."""
    candidate_to_score = {}
    for candidate in tqdm(candidates, desc="Evaluating candidates"):
        if candidate not in utilities:
            utilities[candidate] = {}

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_threads
            ) as executor:
                future_to_agent_id = {
                    executor.submit(agent.get_approval, candidate): agent.get_id()
                    for agent in unmatched_agents
                }
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_agent_id),
                    total=len(future_to_agent_id),
                ):
                    approval, query2_logs = future.result()
                    agent_id = future_to_agent_id[future]
                    pd.DataFrame(query2_logs, columns=LOGS_COLUMNS).to_csv(
                        logs_path, mode="a", header=False
                    )
                    utilities[candidate][agent_id] = approval

            # After all threads finished, do io stuff
            info_df = pd.read_csv(info_path, index_col=0)
            for agent in unmatched_agents:
                agent_id = agent.get_id()
                info_df.loc[candidate, f"approval_{agent_id}"] = utilities[candidate][
                    agent_id
                ]
            info_df.to_csv(info_path)

        sorted_utilities = sorted(
            [utilities[candidate][agent.get_id()] for agent in unmatched_agents],
            reverse=True,
        )
        if len(sorted_utilities) >= coalition_size:
            score = sorted_utilities[coalition_size - 1]
        else:
            score = sorted_utilities[-1]
        candidate_to_score[candidate] = score

    return utilities, candidate_to_score


def generate_slate_ensemble_greedy(
    *,
    agents,
    generators,
    slate_size: int,
    id_to_fixed_statements: dict[str, str] | None = None,
    candidates_persist: bool = False,
    full_log_dir: Path | None = None,
    verbose: bool = False,
    num_threads: int = 1,  # if =1, then serial, if >1, multithreading
) -> tuple[list[str], dict, dict]:
    """
    Generate a slate of statements using ensemble greedy algorithm.

    agents: list of agents
    generators: list of generators
    slate_size : slate size
    id_to_fixed_statements: dict mapping str id of fixed statement (e.g. EditGenerator output -- something that doesn't depend on path of greedy algo) to statement
    candidates_persist: if True, then greedy algorithm picks from pool of ALL candidates generated so far. if False, then greedy algorithm only picks from candidates generated this round, plus the fixed statements
    full_log_dir: directory name to save logs (if exists and already has logs.csv, then logs will be appended)
    verbose
    num_threads: for LLM call multithreading
    """

    # TODO:
    # - what if duplicate candidates? no idea if this breaks logging

    k = slate_size
    n = len(agents)

    # Initialize logging
    full_log_dir, logs_path, info_path = initialize_logging(full_log_dir, k, n)
    # logs (logs.csv): are raw LLM logs (LLMLog field type, maybe with some extra cols)
    # infos (info.csv): have cols source, round_added, approval_{agent_id} for all agents

    assert len(set(agent.get_id() for agent in agents)) == n, "Agent IDs must be unique"
    agents_round_matched = {agent.get_id(): None for agent in agents}

    utilities = {}  # utilities[candidate][agent_id] = approval
    slate = []
    slate_sources = []
    slate_utilities = {}
    if id_to_fixed_statements is None:
        id_to_fixed_statements = dict()
    candidates = set(id_to_fixed_statements.values())  # set of candidate statements

    # Add initial_statements to candidates & info.csv

    ### First: do some initial filtering of id_to_fixed_statements
    for statement_id, statement in id_to_fixed_statements.items():
        pd.DataFrame(
            [
                {
                    "candidate": statement,
                    "round_added": -1,
                    "source": statement_id,
                }
            ]
        ).to_csv(info_path, mode="a", header=False, index=0)

    ## Main algorithm loop
    for round_num in range(k):
        if not candidates_persist:
            # Only stuff to repeat every round is the fixed statements
            candidates = set(id_to_fixed_statements.values()).difference(slate)
        else:
            # Repeat everything every round except stuff that was selected from slate
            candidates = candidates.difference(slate)

        if verbose:
            print(f"Step {round_num + 1} / {k}")

        # Find unmatched agents
        unmatched_agents = [
            agent for agent in agents if agents_round_matched[agent.get_id()] is None
        ]

        # Generate other statements (& update logs.csv, info.csv)
        generated_candidates = generate_statements(
            generators=generators,
            unmatched_agents=unmatched_agents,
            logs_path=logs_path,
            info_path=info_path,
            round_num=round_num,
            num_threads=num_threads,
        )

        candidates.update(generated_candidates)
        assert len(candidates) >= 1

        j = round_num + 1
        # If n divides k, then coalition_size = n//k always. Otherwise, sometimes it's ceil(n/k) and sometimes floor(n/k),
        # so that in the end all n agents are matched to a candidate. See paper for details.
        coalition_size = math.ceil(n / k) if j <= n - k * (n // k) else n // k

        # Evaluate candidates
        utilities, candidate_to_score = evaluate_candidates(
            unmatched_agents=unmatched_agents,
            candidates=candidates,
            utilities=utilities,
            logs_path=logs_path,
            info_path=info_path,
            num_threads=num_threads,
            coalition_size=coalition_size,
        )

        # Find candidate that gets highest score (which is coalition_size'th highest utility, computed in evaluate_candidates)
        best_candidate = max(candidate_to_score, key=candidate_to_score.get)
        # Save to info.csv
        info_df = pd.read_csv(info_path, index_col=0)
        info_df.loc[best_candidate, "round_chosen"] = round_num
        best_candidate_source = info_df.loc[best_candidate, "source"]
        info_df.to_csv(info_path)

        # Update slate
        slate.append(best_candidate)
        slate_sources.append(best_candidate_source)
        slate_utilities[best_candidate] = utilities[best_candidate]

        unmatched_agent_id_to_best_candiate_utility = {
            agent.get_id(): utilities[best_candidate][agent.get_id()]
            for agent in unmatched_agents
        }
        sorted_unmatched_agents = sorted(
            unmatched_agent_id_to_best_candiate_utility,
            key=unmatched_agent_id_to_best_candiate_utility.get,
            reverse=True,
        )

        info_df = pd.read_csv(info_path, index_col=0)
        for agent_id in sorted_unmatched_agents[:coalition_size]:
            agents_round_matched[agent_id] = round_num
            info_df.loc[best_candidate, "matched_" + str(agent_id)] = 1  # true
        info_df.to_csv(info_path)

    with open(full_log_dir / "committee.txt", "w") as f:
        f.write(str(slate_utilities) + "\n")
        # For easy python copy paste
        f.write(str(slate) + "\n")
        f.write("\n\n")
        # For easy markdown copy paste
        f.write(
            "\n".join(
                [
                    f"- ({source}) {statement}"
                    for source, statement in zip(slate_sources, slate)
                ]
            )
        )

    return slate, agents_round_matched, slate_utilities
