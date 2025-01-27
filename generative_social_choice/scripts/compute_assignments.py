import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from generative_social_choice.utils.helper_functions import get_base_dir_path
from generative_social_choice.slates.survey_assignments import AssignmentResult, compute_assignments
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
ASSIGNMENT_FILE = get_base_dir_path() / "data/demo_data/assignments.json"


def run(
        slate_size: int,
        voting_algotirhm: VotingAlgorithm,
        utility_matrix_file: Path,
        statement_id_file: Path,
        assignment_file: Optional[Path]=None,
    ):
    result = compute_assignments(
        voting_algorithm=voting_algotirhm,
        utility_matrix_file=utility_matrix_file,
        statement_id_file=statement_id_file,
        slate_size=slate_size,
    )
    if assignment_file is not None:
        print("Storing results ...")
        result.save(assignment_file)

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
        "--statement_id_file",
        type=Path,
        default=STATEMENT_ID_FILE,
        help="Path to the file containing the mapping from statement ID to statement.",
    )

    parser.add_argument(
        "--assignment_file",
        type=Path,
        default=ASSIGNMENT_FILE,
        help="The computed solution will be written to this path.",
    )

    parser.add_argument(
        "--voting_algorithm",
        type=str,
        default="phragmen",
        help="Name of the voting algorithm to use.",
    )

    args = parser.parse_args()

    if args.voting_algorithm.lower() in ["phragmen", "sequentialphragmenminimax"]:
        voting_algorithm = SequentialPhragmenMinimax()
    elif args.voting_algorithm.lower() in ["exact", "exacttotalutilitymaximization"]:
        voting_algorithm = ExactTotalUtilityMaximization()
    elif args.voting_algorithm.lower() in ["greedy", "greedytotalutilitymaximization"]:
        voting_algorithm = GreedyTotalUtilityMaximization()
    elif args.voting_algorithm.lower() in ["lp", "lptotalutilitymaximization"]:
        voting_algorithm = LPTotalUtilityMaximization()
    else:
        raise ValueError("Invalid algorithm name passed!")
    #TODO Might want to pass some other arguments as well
    # (or alternatively, by default create different assignments for a list of algorithms)

    run(
        slate_size=args.slate_size,
        voting_algotirhm=voting_algorithm,
        utility_matrix_file=args.utility_matrix_file,
        statement_id_file=args.statement_id_file,
        assignment_file=args.assignment_file,
    )