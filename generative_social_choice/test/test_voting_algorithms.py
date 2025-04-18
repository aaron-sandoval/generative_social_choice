from pathlib import Path
import re
import unittest
from typing import Optional, Sequence, Generator, Hashable, override
from dataclasses import dataclass
import sys
import inspect
import multiprocessing
from functools import partial
import itertools
import pandas as pd
import numpy as np
from parameterized import parameterized
from kiwiutils.finite_valued import all_instances
from kiwiutils.kiwilib import leafClasses

# Add the project root directory to the system path
sys.path.append(str(Path(__file__).parent.parent.parent))

from generative_social_choice.slates.voting_algorithms import (
    SequentialPhragmenMinimax,
    GreedyTotalUtilityMaximization,
    ExactTotalUtilityMaximization,
    LPTotalUtilityMaximization,
    VotingAlgorithm,
    GeometricTransformation,
)
from generative_social_choice.slates.voting_algorithm_axioms import (
    IndividualParetoAxiom,
    HappiestParetoAxiom,
    CoverageAxiom,
    MinimumAndTotalUtilityParetoAxiom,
    VotingAlgorithmAxiom,
)
from generative_social_choice.utils.helper_functions import (
    get_time_string,
    get_base_dir_path,
    sanitize_name,
)
from generative_social_choice.test.utilities_for_testing import rated_vote_cases, RatedVoteCase

_NAME_DELIMITER = "$&$"

# Instances of voting algorithms to test, please add more as needed
# voting_algorithms_to_test: Generator[VotingAlgorithm, None, None] = all_instances(VotingAlgorithm)
voting_algorithms_to_test = (
    GreedyTotalUtilityMaximization(),
    ExactTotalUtilityMaximization(),
    LPTotalUtilityMaximization(),
    GreedyTotalUtilityMaximization(utility_transform=GeometricTransformation(p=1.5)),
    ExactTotalUtilityMaximization(utility_transform=GeometricTransformation(p=1.5)),
    LPTotalUtilityMaximization(utility_transform=GeometricTransformation(p=1.5)),
    *all_instances(SequentialPhragmenMinimax),
)

voting_algorithm_test_cases: tuple[tuple[str, VotingAlgorithm, RatedVoteCase], ...] = tuple((algo.name + "___" + rated.name, rated, algo) for rated, algo in itertools.product(rated_vote_cases.values(), voting_algorithms_to_test))

axioms_to_evaluate: tuple[VotingAlgorithmAxiom, ...] = tuple(axiom() for axiom in filter(lambda x: not inspect.isabstract(x), sorted(leafClasses(VotingAlgorithmAxiom), key=lambda x: x.__name__)))

class AlgorithmEvaluationResult(unittest.TestResult):
    """
    Custom TestResult class to log test results into a DataFrame and write to CSV.
    """
    included_subtests: tuple[VotingAlgorithmAxiom, ...] = axioms_to_evaluate
    log_filename: Path = get_base_dir_path() / "data" / "voting_algorithm_evals" / f"{get_time_string()}.csv"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        col_index = pd.MultiIndex.from_product([rated_vote_cases.keys(), [axiom.name for axiom in self.included_subtests]], names=["vote", "subtest"])
        self.results = pd.DataFrame(index=[algo.name for algo in voting_algorithms_to_test], columns=col_index)

    @override
    def addSubTest(self, test, subtest, outcome):
        super().addSubTest(test, subtest, outcome)
        # if subtest._message not in self.included_subtests:
        #     return
        if outcome is not None:
            a = 1 # DEBUG
        alg_name, vote_name, subtest_name = subtest._message.split(_NAME_DELIMITER)
        if not pd.isna(self.results.at[alg_name, (vote_name, subtest_name)]):
            raise ValueError(f"Result already exists for {alg_name}, {vote_name}, {subtest_name}")
        self.results.at[alg_name, (vote_name, subtest_name)] = 1 if outcome is None else 0

    def stopTestRun(self):
        self.write_to_csv()

    def write_to_csv(self):
        self.results.to_csv(self.log_filename, index=True)


class TestVotingAlgorithmFunctionality(unittest.TestCase):
    """
    Test the functionality and properties of voting algorithms.
    """
    @parameterized.expand(voting_algorithm_test_cases)
    def test_voting_algorithm_functionality(
        self,
        name: str,
        rated_vote_case: RatedVoteCase,
        voting_algorithm: VotingAlgorithm,
    ):
        """
        Test the sequential Phragmen Minimax algorithm for rated voting.

        # Arguments
        
        """
        # Compute the solution using the voting algorithm
        slate, assignments = voting_algorithm.vote(
            rated_vote_case.rated_votes.copy(),  # Voting algorithms might append columns
            rated_vote_case.slate_size,
        )

        # Basic functionality and result format
        with self.subTest(msg="A Basic functionality"):
            self.assertEqual(len(slate), rated_vote_case.slate_size)
            self.assertEqual(len(set(slate)), len(slate))
            self.assertEqual(len(assignments), len(rated_vote_case.rated_votes))
            self.assertLessEqual(set(assignments.candidate_id), set(slate))

        # TODO: These types of tests will move to separate test cases for each algorithm
        # Check that the assignments are valid. For functional debugging only, will be omitted from algorithm evaluation
        # if rated_vote_case.expected_assignments is not None:
        #     with self.subTest(msg="B Assignments"):
        #         assert pd.DataFrame.equals(assignments.candidate_id, rated_vote_case.expected_assignments.candidate_id)

        #     with self.subTest(msg="C Assignments other columns"):
        #         for col in ["utility", "load", "utility_previous", "second_selected_candidate_id"]:
        #             if col in rated_vote_case.expected_assignments.columns:
        #                 assert pd.DataFrame.equals(assignments[col], rated_vote_case.expected_assignments[col])


def run_single_axiom_test(test_case):
    """
    Run a single axiom test case in a separate process.
    
    Args:
        test_case: A tuple containing (voting_algorithm, rated_vote_case, axiom)
        
    Returns:
        A tuple containing (voting_algorithm_name, rated_vote_case_name, axiom_name, success)
    """
    voting_algorithm, rated_vote_case, axiom = test_case
    
    # Check if the axiom is satisfied for all augmented cases
    for rated_votes in rated_vote_case.augmented_cases:
        # Compute the solution using the voting algorithm
        slate, assignments = voting_algorithm.vote(
            rated_votes.copy(),  # Voting algorithms might append columns
            rated_vote_case.slate_size,
        )
        
        # Use the return value of evaluate_assignment directly
        if not axiom.evaluate_assignment(
            rated_votes=rated_votes, 
            slate_size=rated_vote_case.slate_size, 
            assignments=assignments
        ):
            return (voting_algorithm.name, rated_vote_case.name, axiom.name, False)
    
    # If we get here, all augmented cases passed the axiom test
    return (voting_algorithm.name, rated_vote_case.name, axiom.name, True)


class TestVotingAlgorithmAgainstAxioms(unittest.TestCase):
    """
    Test whether voting algorithms satisfy various axioms using parallel processing.
    """
    
    def setUp(self):
        """Set up the test environment."""
        # Create a pool of worker processes
        self.process_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        
    def tearDown(self):
        """Clean up resources after tests."""
        self.process_pool.close()
        self.process_pool.join()
    
    def test_voting_algorithm_for_pareto(self):
        """
        Test whether the algorithms satisfy axioms using parallel processing.
        """
        # Create all test cases
        test_cases = []
        for voting_algorithm in voting_algorithms_to_test:
            for rated_vote_case in rated_vote_cases.values():
                for axiom in axioms_to_evaluate:
                    test_cases.append((voting_algorithm, rated_vote_case, axiom))
        
        # Run tests in parallel
        results = self.process_pool.map(run_single_axiom_test, test_cases)
        
        # Process results
        for voting_algorithm_name, rated_vote_case_name, axiom_name, success in results:
            with self.subTest(msg=_NAME_DELIMITER.join([voting_algorithm_name, rated_vote_case_name, axiom_name])):
                self.assertTrue(success, f"Algorithm {voting_algorithm_name} failed axiom {axiom_name} for vote case {rated_vote_case_name}")


class TestVotingAlgorithmAssignments(unittest.TestCase):

    @parameterized.expand([
        (rated_vote_cases["Ex 1.2"], ["s2", "s3"], None, {}),
        (rated_vote_cases["Ex 1.2"], ["s2", "s3"], None, dict(load_magnitude_method="total")),
        (rated_vote_cases["Ex 1.1"], ["s2", "s4"], None, {}),
        (rated_vote_cases["Ex 1.1"], ["s2", "s4"], None, dict(load_magnitude_method="total")),
        (rated_vote_cases["Ex 1.1 modified"], ["s2", "s4"], None, {}),
        (rated_vote_cases["Ex 1.1 modified"], ["s2", "s4"], None, dict(load_magnitude_method="total")),
        # (rated_vote_cases["Ex Alg A.1"], ["s1", "s3", "s4"], None, {}), # Stochastic selection
    ])
    def test_phragmen_assignments(
        self,
        rated_vote_case: RatedVoteCase,
        expected_slate: Optional[list[str]] = None,
        expected_assignments: Optional[list[str]] = None,
        algorithm_kwargs: Optional[dict]={},
    ):
        
        slate, assignments = SequentialPhragmenMinimax(**algorithm_kwargs).vote(
            rated_vote_case.rated_votes.copy(),  # Voting algorithms might append columns
            rated_vote_case.slate_size,
        )

        if expected_slate is not None:
            expected_slate = frozenset(expected_slate)
            self.assertEqual(set(slate), expected_slate)
        if expected_assignments is not None:
            expected_assignments = pd.DataFrame({"candidate_id": expected_assignments}, index=rated_vote_case.rated_votes.index)

        pass

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestVotingAlgorithmAgainstAxioms)
    runner = unittest.TextTestRunner(resultclass=AlgorithmEvaluationResult)
    runner.run(suite)
