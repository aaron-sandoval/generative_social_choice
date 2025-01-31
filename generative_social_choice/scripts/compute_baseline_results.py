import random
import argparse
from typing import Optional
from pathlib import Path

import pandas as pd
import numpy as np

from generative_social_choice.utils.helper_functions import get_base_dir_path
from generative_social_choice.ratings.rating_generation import get_agents
from generative_social_choice.ratings.utility_matrix import create_utility_matrix
from generative_social_choice.paper_replication.compute_matching import optimize_monroe_matching
from generative_social_choice.slates.survey_assignments import AssignmentResult


STATEMENTS_FILE = get_base_dir_path() / "data/ratings_and_matching.csv"  # column names include statements

RATINGS_FILE = get_base_dir_path() / "data/demo_data/baseline_ratings.jsonl"
LOG_FILE = get_base_dir_path() / "data/demo_data/baseline_ratings_logs.csv"

UTILITY_MATRIX_FILE = get_base_dir_path() / "data/demo_data/baseline_utility_matrix.csv"
STATEMENT_ID_FILE = get_base_dir_path() / "data/demo_data/baseline_utility_matrix_statements.csv"

ASSIGNMENT_FILE = get_base_dir_path() / "data/demo_data/baseline_assignments.json"


def create_matrix(model: str, verbose: bool=True, num_agents: Optional[int]=None, num_statements: Optional[int]=None):
    agents = get_agents(model=model)

    # Subsample agents
    if num_agents is not None:
        agents = random.sample(agents, num_agents)

    # Read generated statements
    statements = [col for col in pd.read_csv(STATEMENTS_FILE).columns.to_list() if len(col)>50]

    if verbose:
        print(f"Read {len(statements)} statements from file.")

    # Subsample statements
    if num_statements is not None:
        statements = random.sample(statements, num_statements)

    create_utility_matrix(
        agents=agents,
        statements=statements,
        utility_matrix_file=UTILITY_MATRIX_FILE,
        statement_id_file=STATEMENT_ID_FILE,
        ratings_file=RATINGS_FILE,
        prepend_survey_statements=False,
        log_file=LOG_FILE,
        verbose=verbose,
    )

def compute_baseline_assignments(
        utility_matrix_file: Path,
        statement_id_file: Path,
    ) -> AssignmentResult:
    utilities_df = pd.read_csv(utility_matrix_file, index_col=0)
    statement_df = pd.read_csv(statement_id_file, index_col=0)

    # Utility function to get statements based on IDs (column names in utility df)
    id_to_statement = lambda statement: statement_df.loc[statement]["statement"]

    slate = utilities_df.columns.to_list()

    # Write utilities matrix
    utilities = []
    for agent_id in utilities_df.index:
        utilities.append(utilities_df.loc[agent_id].to_list())
    assignment_ixs = optimize_monroe_matching(utilities=utilities)
    # This returns a list of integer indices starting from 0 for the first statements

    assignments = [slate[i] for i in assignment_ixs]

    # Convert to AssignmentResult format
    agent_ids = utilities_df.index.to_list()
    utilities = [float(utilities_df.loc[agent_ids[ix], assignment]) for ix, assignment in enumerate(assignments)]
    result = AssignmentResult(
        slate=slate,
        slate_statements=[id_to_statement(statement_id) for statement_id in slate],
        agent_ids=agent_ids,
        assignments=assignments,
        utilities=utilities,
        info={"type": "baseline"},
    )

    return result


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_agents",
        type=int,
        default=None,
        help="Number of agents to consider in the utility matrix. If not provided, all agents will be used.",
    )

    parser.add_argument(
        "--num_statements",
        type=int,
        default=None,
        help="Number of statements to consider in the utility matrix. If not provided, all statements will be used.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Default is gpt-4o. Fish's experiments (late 2023) used gpt-4-32k-0613 (publicly unavailable).",
    )

    args = parser.parse_args()

    create_matrix(model=args.model, num_statements=args.num_statements, num_agents=args.num_agents)

    # Now assign voters to statements using Fish et al.'s method
    result = compute_baseline_assignments(utility_matrix_file=UTILITY_MATRIX_FILE, statement_id_file=STATEMENT_ID_FILE)
    if ASSIGNMENT_FILE is not None:
        print("Storing results ...")
        result.save(ASSIGNMENT_FILE)

    print("\nRESULT\n")

    print("Slate found by algorithm:", result.slate)
    for statement_id, statement in zip(result.slate, result.slate_statements):
        print(f"{statement_id}: {statement.replace("\n", " ")}")

    print("\nAssignments:")
    for statement_id in result.slate:
        num_assigned = sum(np.array(result.assignments)==statement_id)
        print(f"{statement_id}: {num_assigned}")

    print("\nTotal utility:", sum(result.utilities))

    print("\nUtilities")
    print(result.utilities)