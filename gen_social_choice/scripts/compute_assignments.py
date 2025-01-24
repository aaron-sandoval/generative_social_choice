import numpy as np

from gen_social_choice.utils.helper_functions import get_base_dir_path
from gen_social_choice.slates.survey_assignments import AssignmentResult, compute_assignments
from gen_social_choice.slates.voting_algorithms import (
    SequentialPhragmenMinimax,
    GreedyTotalUtilityMaximization,
)


# Input
UTILITY_MATRIX_FILE = get_base_dir_path() / "data/demo_data/TEST_utility_matrix.csv"
STATEMENT_ID_FILE = get_base_dir_path() / "data/demo_data/TEST_utility_matrix_statements.csv"

# Output
ASSIGNMENT_FILE = get_base_dir_path() / "data/demo_data/TEST_assignments.json"


def run():
    result = compute_assignments(
        #voting_algorithm=SequentialPhragmenMinimax(),
        voting_algorithm=GreedyTotalUtilityMaximization(),
        utility_matrix_file=UTILITY_MATRIX_FILE,
        statement_id_file=STATEMENT_ID_FILE,
        slate_size=3,
    )
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

    print("\nUtilities")
    print(result.utilities)


if __name__=="__main__":
    run()