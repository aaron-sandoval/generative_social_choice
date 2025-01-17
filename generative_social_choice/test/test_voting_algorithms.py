from pathlib import Path
import re
import unittest
from typing import Optional, Sequence, Generator, Hashable, override
from dataclasses import dataclass
import sys

import itertools
import pandas as pd
import numpy as np
from parameterized import parameterized
from kiwiutils.finite_valued import all_instances
from kiwiutils.kiwilib import getAllSubclasses

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
from generative_social_choice.utils.helper_functions import get_time_string, get_base_dir_path
from generative_social_choice.test.utilities_for_testing import rated_vote_cases, RatedVoteCase

# Instances of voting algorithms to test, please add more as needed
# voting_algorithms_to_test: Generator[VotingAlgorithm, None, None] = all_instances(VotingAlgorithm)
voting_algorithms_to_test = (
    GreedyTotalUtilityMaximization(),
    ExactTotalUtilityMaximization(),
    LPTotalUtilityMaximization(),
    GreedyTotalUtilityMaximization(utility_transform=GeometricTransformation(p=1.5)),
    ExactTotalUtilityMaximization(utility_transform=GeometricTransformation(p=1.5)),
    LPTotalUtilityMaximization(utility_transform=GeometricTransformation(p=1.5)),
    SequentialPhragmenMinimax(),
    SequentialPhragmenMinimax(load_magnitude_method="total"),
)

voting_algorithm_test_cases: tuple[tuple[str, VotingAlgorithm, RatedVoteCase], ...] = tuple((algo.name + "___" + rated.name, rated, algo) for rated, algo in itertools.product(rated_vote_cases.values(), voting_algorithms_to_test))

axioms_to_evaluate: tuple[VotingAlgorithmAxiom, ...] = tuple(axiom() for axiom in getAllSubclasses(VotingAlgorithmAxiom))

class AlgorithmEvaluationResult(unittest.TestResult):
    """
    Custom TestResult class to log test results into a DataFrame and write to CSV.
    """
    included_subtests: set[str] = set(axioms_to_evaluate)
    log_filename: Path = get_base_dir_path() / "data" / "voting_algorithm_evals" / f"{get_time_string()}.csv"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        col_index = pd.MultiIndex.from_product([rated_vote_cases.keys(), self.included_subtests], names=["vote", "subtest"])
        self.results = pd.DataFrame(index=[algo.name for algo in voting_algorithms_to_test], columns=col_index)

    @override
    def addSubTest(self, test, subtest, outcome):
        super().addSubTest(test, subtest, outcome)
        if subtest._message not in self.included_subtests:
            return
        if outcome is not None:
            a = 1 # DEBUG
        alg_name, vote_name = repr(subtest.test_case).split("___")
        vote_name = vote_name[:-1]
        alg_name = re.sub(r'^.*?_[0-9]+_', '', alg_name)
        subtest_name = subtest._message
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

        # TODO: These types of tests will move to separate test cases for each algorithm
        # Check that the assignments are valid. For functional debugging only, will be omitted from algorithm evaluation
        # if rated_vote_case.expected_assignments is not None:
        #     with self.subTest(msg="B Assignments"):
        #         assert pd.DataFrame.equals(assignments.candidate_id, rated_vote_case.expected_assignments.candidate_id)

        #     with self.subTest(msg="C Assignments other columns"):
        #         for col in ["utility", "load", "utility_previous", "second_selected_candidate_id"]:
        #             if col in rated_vote_case.expected_assignments.columns:
        #                 assert pd.DataFrame.equals(assignments[col], rated_vote_case.expected_assignments[col])


class TestVotingAlgorithmParetoAxioms(unittest.TestCase):
    @parameterized.expand(voting_algorithm_test_cases)
    def test_voting_algorithm_for_pareto(
        self,
        name: str,
        rated_vote_case: RatedVoteCase,
        voting_algorithm: VotingAlgorithm,
    ):
        """
        Test whether the algorithm satisfies various forms of Pareto efficiency.

        # Arguments
        
        """
        # Compute the solution using the voting algorithm
        W, assignments = voting_algorithm.vote(
            rated_vote_case.rated_votes.copy(),  # Voting algorithms might append columns
            rated_vote_case.slate_size,
        )

        for axiom in axioms_to_evaluate:
            with self.subTest(msg=axiom.name):
                assert axiom.evaluate_assignment(rated_votes=rated_vote_case.rated_votes, slate_size=rated_vote_case.slate_size, assignments=assignments), \
                    f"{axiom.name} is not satisfied"

        # with self.subTest(msg=axioms_to_evaluate[0]):
        #     axiom = MinimumAndTotalUtilityParetoAxiom()
        #     assert frozenset(W) in axiom.satisfactory_slates(rated_vote_case.rated_votes, rated_vote_case.slate_size), "The selected slate is not among the Pareto efficient slates"

        # # TODO: make this a function in voting_utils
        # if rated_vote_case.non_extremal_pareto_efficient_slates is not None:
        #     with self.subTest(msg=axioms_to_evaluate[1]):
        #         assert frozenset(W) in {frozenset(pareto_slate) for pareto_slate in rated_vote_case.non_extremal_pareto_efficient_slates}, "The selected slate is not among the non-extremal Pareto efficient slates"

        # with self.subTest(msg=axioms_to_evaluate[2]):
        #     axiom = IndividualParetoAxiom()
        #     assert axiom.evaluate_assignment(rated_votes=rated_vote_case.rated_votes, slate_size=rated_vote_case.slate_size, assignments=assignments), \
        #         "There is a slate with strictly greater total utility and no lesser utility for any individual member"

        # with self.subTest(msg=axioms_to_evaluate[3]):
        #     axiom = HappiestParetoAxiom()
        #     assert axiom.evaluate_assignment(rated_votes=rated_vote_case.rated_votes, slate_size=rated_vote_case.slate_size, assignments=assignments), \
        #         "There is a slate with a strictly better m-th happiest person curve"

        # with self.subTest(msg=axioms_to_evaluate[4]):
        #     axiom = CoverageAxiom()
        #     assert axiom.evaluate_assignment(rated_votes=rated_vote_case.rated_votes, slate_size=rated_vote_case.slate_size, assignments=assignments), \
        #         "There is a slate which represents more people"


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestVotingAlgorithms)
    runner = unittest.TextTestRunner(resultclass=AlgorithmEvaluationResult)
    runner.run(suite)
