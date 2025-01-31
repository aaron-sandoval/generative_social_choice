import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np

from generative_social_choice.utils.helper_functions import get_base_dir_path
from generative_social_choice.slates.survey_assignments import compute_assignments
from generative_social_choice.slates.voting_algorithms import (
    VotingAlgorithm,
    SequentialPhragmenMinimax,
    GreedyTotalUtilityMaximization,
    ExactTotalUtilityMaximization,
    LPTotalUtilityMaximization,
)


# Input
UTILITY_MATRIX_FILE = get_base_dir_path() / "data/demo_data/utility_matrix.csv"
STATEMENT_ID_FILE = get_base_dir_path() / "data/demo_data/utility_matrix_statements.csv"

# Output
ASSIGNMENT_DIR = get_base_dir_path() / "data/demo_data/assignments/"


def run(
        slate_size: int,
        voting_algotirhm: VotingAlgorithm,
        utility_matrix_file: Path,
        statement_id_file: Path,
        assignment_file: Optional[Path]=None,
        ignore_initial_statements: bool=False,
        verbose: bool=False,
    ):
    result = compute_assignments(
        voting_algorithm=voting_algotirhm,
        utility_matrix_file=utility_matrix_file,
        statement_id_file=statement_id_file,
        slate_size=slate_size,
        ignore_initial_statements=6 if ignore_initial_statements else None,
    )
    if assignment_file is not None:
        if verbose:
            print("Storing results ...")
        result.save(assignment_file)

    if verbose:
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
    return result


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--slate_size",
        type=int,
        required=True,
        help="Number of statements to include in the slate.",
    )

    parser.add_argument(
        "--utility_matrix_file",
        type=Path,
        default=UTILITY_MATRIX_FILE,
        help="Path to the file containing the utility matrix.",
    )

    parser.add_argument(
        "--ignore_initial",
        type=bool,
        default=False,
        help="If True, the first 6 statements in the utility matrix will be ignored.",
    )

    parser.add_argument(
        "--statement_id_file",
        type=Path,
        default=STATEMENT_ID_FILE,
        help="Path to the file containing the mapping from statement ID to statement.",
    )

    parser.add_argument(
        "--assignment_dir",
        type=Path,
        default=ASSIGNMENT_DIR,
        help="The computed solutions will be written to this directory.",
    )

    args = parser.parse_args()

    if not args.assignment_dir.exists():
        print(f"Creating directory {args.assignment_dir} ...")
        os.makedirs(args.assignment_dir)

    # Keys will be used as filenames
    voting_algorithms = {
        "phragmen": SequentialPhragmenMinimax(),
        "exact": ExactTotalUtilityMaximization(),
        "greedy": GreedyTotalUtilityMaximization(),
        "lp": LPTotalUtilityMaximization(),
    }
    for name, algo in voting_algorithms.items():
        print(f"\n\nRunning algorithm '{algo.name}' ...")
        result = run(
            slate_size=args.slate_size,
            voting_algotirhm=algo,
            utility_matrix_file=args.utility_matrix_file,
            statement_id_file=args.statement_id_file,
            assignment_file=args.assignment_dir / f"{name}.json",
            ignore_initial_statements=args.ignore_initial,
            verbose=True,
        )