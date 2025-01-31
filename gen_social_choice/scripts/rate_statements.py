import random
import argparse
from typing import Optional

import pandas as pd

from gen_social_choice.utils.helper_functions import get_base_dir_path
from gen_social_choice.ratings.rating_generation import get_agents
from gen_social_choice.ratings.utility_matrix import create_utility_matrix


#STATEMENTS_FILE = get_base_dir_path() / "data/demo_data/2025-01-27-112106_statement_generation/statement_generation_raw_output.csv"
STATEMENTS_FILE = get_base_dir_path() / "data/demo_data/statement_generation_selection.csv"

RATINGS_FILE = get_base_dir_path() / "data/demo_data/ratings.jsonl"
LOG_FILE = get_base_dir_path() / "data/demo_data/ratings_logs.csv"

UTILITY_MATRIX_FILE = get_base_dir_path() / "data/demo_data/utility_matrix.csv"
STATEMENT_ID_FILE = get_base_dir_path() / "data/demo_data/utility_matrix_statements.csv"


def run(model: str, verbose: bool=True, num_agents: Optional[int]=None, num_statements: Optional[int]=None):
    agents = get_agents(model=model)

    # Subsample agents
    if num_agents is not None:
        agents = random.sample(agents, num_agents)

    # Read generated statements
    statements = pd.read_csv(STATEMENTS_FILE)["statement"].to_list()

    # Subsample statements
    if num_statements is not None:
        statements = random.sample(statements, num_statements)

    create_utility_matrix(
        agents=agents,
        statements=statements,
        utility_matrix_file=UTILITY_MATRIX_FILE,
        statement_id_file=STATEMENT_ID_FILE,
        ratings_file=RATINGS_FILE,
        prepend_survey_statements=True,
        log_file=LOG_FILE,
        verbose=verbose,
    )


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
        help="Number of statements to consider in the utility matrix. If not provided, all agents will be used.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Default is gpt-4o-mini. Fish's experiments (late 2023) used gpt-4-32k-0613 (publicly unavailable).",
    )

    args = parser.parse_args()

    run(model=args.model, num_statements=args.num_statements, num_agents=args.num_agents)