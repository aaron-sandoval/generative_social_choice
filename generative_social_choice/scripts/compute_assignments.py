import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
from kiwiutils.finite_valued import all_instances

from generative_social_choice.utils.helper_functions import get_results_paths
from generative_social_choice.slates.survey_assignments import compute_assignments
from generative_social_choice.slates.voting_algorithms import (
    VotingAlgorithm,
    SequentialPhragmenMinimax,
    GreedyTotalUtilityMaximization,
    ExactTotalUtilityMaximization,
    LPTotalUtilityMaximization,
    GeometricTransformation,
)


VOTING_ALGORITHMS = {
    **{alg.name: alg for alg in all_instances(SequentialPhragmenMinimax)},
    "exact": ExactTotalUtilityMaximization(),
    "greedy": GreedyTotalUtilityMaximization(),
    "lp": LPTotalUtilityMaximization(),
    "greedy (p=1.5)": GreedyTotalUtilityMaximization(utility_transform=GeometricTransformation(p=1.5)),
    "exact (p=1.5)": ExactTotalUtilityMaximization(utility_transform=GeometricTransformation(p=1.5)),
    "lp (p=1.5)": LPTotalUtilityMaximization(utility_transform=GeometricTransformation(p=1.5)),
}

def run(
        slate_size: int,
        voting_algotirhm: VotingAlgorithm,
        utility_matrix_file: Path,
        statement_id_file: Path,
        assignment_file: Optional[Path]=None,
        include_seed: bool = False,
        verbose: bool = False,
    ):
    # By default we exclude the first 6 (seed) statements. With include_seed, append "_with-seed" to filename.
    ignore_initial_statements = None if include_seed else 6
    if assignment_file is not None and include_seed:
        assignment_file = assignment_file.parent / (assignment_file.stem + "_with-seed" + assignment_file.suffix)
    if assignment_file is not None and not assignment_file.parent.exists():
        print(f"Creating directory {assignment_file.parent} ...")
        assignment_file.parent.mkdir(parents=True, exist_ok=True)

    result = compute_assignments(
        voting_algorithm=voting_algotirhm,
        utility_matrix_file=utility_matrix_file,
        statement_id_file=statement_id_file,
        slate_size=slate_size,
        ignore_initial_statements=ignore_initial_statements,
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
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model used for labelling. Default is gpt-4o-mini.",
    )

    parser.add_argument(
        "--embedding_type",
        type=str,
        choices=["llm", "seed_statement", "fish"],
        default="llm",
        help="Type of embeddings used. Default is llm.",
    )

    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional run ID to use for organizing results in a specific directory.",
    )

    parser.add_argument(
        "--include_seed",
        action="store_true",
        help="If set, the first 6 (seed) statements are included in the candidate set. By default only generated statements are considered. Results are saved with '_with-seed' in the filename.",
    )

    args = parser.parse_args()

    # Get paths based on run_id and model
    result_paths = get_results_paths(
        labelling_model="4o-mini" if "mini" in args.model else "4o",
        embedding_type=args.embedding_type,
        baseline=False,
        run_id=args.run_id
    )

    # Keys will be used as filenames
    for name, algo in VOTING_ALGORITHMS.items():
        print(f"\n\nRunning algorithm '{algo.name}' ...")
        result = run(
            slate_size=args.slate_size,
            voting_algotirhm=algo,
            utility_matrix_file=result_paths["utility_matrix_file"],
            statement_id_file=result_paths["statement_id_file"],
            assignment_file=result_paths["assignments"] / f"{name}.json",
            include_seed=args.include_seed,
            verbose=True,
        )