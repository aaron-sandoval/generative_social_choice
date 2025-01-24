import random

import pandas as pd

from gen_social_choice.utils.helper_functions import get_base_dir_path
from gen_social_choice.ratings.rating_generation import get_agents
from gen_social_choice.ratings.utility_matrix import create_utility_matrix


STATEMENTS_FILE = get_base_dir_path() / "data/demo_data/2025-01-22-193423_statement_generation/statement_generation_raw_output.csv"

RATINGS_FILE = get_base_dir_path() / "data/demo_data/TEST_ratings.jsonl"
LOG_FILE = get_base_dir_path() / "data/demo_data/TEST_ratings_logs.csv"

UTILITY_MATRIX_FILE = get_base_dir_path() / "data/demo_data/TEST_utility_matrix.csv"
STATEMENT_ID_FILE = get_base_dir_path() / "data/demo_data/TEST_utility_matrix_statements.csv"


def run(debug_mode: bool=False):
    agents = get_agents(model="gpt-4o-mini")

    # Read generated statements
    statements = pd.read_csv(STATEMENTS_FILE)["statement"].to_list()

    if debug_mode:
        # Subsampling for testing purposes
        #agents = random.sample(agents, 5)
        agents = agents[:5]
        
        # Subsampling for testing purposes
        #statements = random.sample(statements, 2)
        statements = statements[:3]

    create_utility_matrix(
        agents=agents,
        statements=statements,
        utility_matrix_file=UTILITY_MATRIX_FILE,
        statement_id_file=STATEMENT_ID_FILE,
        ratings_file=RATINGS_FILE,
        log_file=LOG_FILE,
        verbose=debug_mode,
    )


if __name__=="__main__":
    run(debug_mode=True)