import random
import argparse
from typing import Optional

import pandas as pd

from generative_social_choice.utils.helper_functions import get_results_paths
from generative_social_choice.ratings.rating_generation import get_agents
from generative_social_choice.ratings.utility_matrix import create_utility_matrix


def run(model: str, verbose: bool=True, num_agents: Optional[int]=None, num_statements: Optional[int]=None, run_id: Optional[str]=None):
    agents = get_agents(model=model)

    # Subsample agents
    if num_agents is not None:
        agents = random.sample(agents, num_agents)

    # Get paths based on run_id
    result_paths = get_results_paths(
        labelling_model="4o-mini" if "mini" in model else "4o",
        embedding_type="llm",  # Default to llm embeddings
        baseline=False,
        run_id=run_id
    )

    # Read generated statements
    statements = pd.read_csv(result_paths["base_dir"] / "statement_generation_raw_output.csv")["statement"].to_list()

    # Subsample statements
    if num_statements is not None:
        statements = random.sample(statements, num_statements)

    create_utility_matrix(
        agents=agents,
        statements=statements,
        utility_matrix_file=result_paths["utility_matrix_file"],
        statement_id_file=result_paths["statement_id_file"],
        ratings_file=result_paths["base_dir"] / "ratings.jsonl",
        prepend_survey_statements=True,
        log_file=result_paths["base_dir"] / "ratings_logs.csv",
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
        help="Number of statements to consider in the utility matrix. If not provided, all statements will be used.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="4o-mini",
        help="Default is 4o-mini. Fish's experiments (late 2023) used gpt-4-32k-0613 (publicly unavailable).",
    )

    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional run ID to use for organizing results in a specific directory.",
    )

    args = parser.parse_args()

    run(model=args.model, num_statements=args.num_statements, num_agents=args.num_agents, run_id=args.run_id)