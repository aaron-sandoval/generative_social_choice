from pathlib import Path
import re
import unittest
from typing import Optional, Sequence, Generator, Hashable, override
from dataclasses import dataclass
import sys

import itertools
import pandas as pd
from parameterized import parameterized

print([p for p in sys.path])
# import generative_social_choice.slates.voting_algorithms
from generative_social_choice.slates.voting_algorithms import (
    SequentialPhragmenMinimax,
    VotingAlgorithm,
)
from generative_social_choice.utils.helper_functions import get_time_string, get_base_dir_path


@dataclass
class RatedVoteCase:
    """
    A voting case with rated votes and expected results.

    # Arguments
    - `rated_votes: pd.DataFrame | list[list[int | float]]`: Utility of each voter (rows) for each candidate (columns)
      - If passed as a nested list, it's converted to a DataFrame with columns named `s1`, `s2`, etc.
    - `slate_size: int`: The number of candidates to be selected
    - `pareto_efficient_slates: Sequence[list[int]]`: Slates that are Pareto efficient on the egalitarian-utilitarian trade-off parameter.
      - Egalitarian objective: Maximize the minimum utility among all individual voters
      - Utilitarian objective: Maximize the total utility among all individual voters
    - `non_extremal_pareto_efficient_slates: Optional[Sequence[list[int]]] = None`: Slates that are non-extremal Pareto efficient on the egalitarian-utilitarian trade-off parameter.
        - Subset of `pareto_efficient_slates` which don't make arbitrarily large egalitarian-utilitarian sacrifices in either direction.
        - Ex: For Example Alg2.1, s1 is Pareto efficient, but not non-extremal Pareto efficient because it makes an arbitrarily large egalitarian sacrifice for an incremental utilitarian gain.
    - `expected_assignments: Optional[pd.DataFrame] = None`: An expected assignment of voters to candidates with the following columns:
        - `candidate_id`: The candidate to which the voter is assigned
        - Other columns not guaranteed to always be present, used for functional testing only.They should always be checked in the unit tests
    """
    rated_votes: pd.DataFrame | list[list[int | float]]
    slate_size: int
    pareto_efficient_slates: set[frozenset[Hashable]]
    non_extremal_pareto_efficient_slates: Optional[set[frozenset[Hashable]]] = None
    expected_assignments: Optional[pd.DataFrame] = None
    name: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.rated_votes, list):
            self.rated_votes = pd.DataFrame(self.rated_votes, columns=[f"s{i}" for i in range(1, len(self.rated_votes) + 1)])

        if self.name is None:
            cols_str = "_".join(str(col) + "_" + "_".join(str(x).replace(".", "p") for x in self.rated_votes[col]) 
                              for col in self.rated_votes.columns)
            self.name = f"k_{self.slate_size}_{cols_str}"
        elif self.name[:2] == "k=":
            self.name = f"k_{self.slate_size}_{self.name}"


# The voting cases to test, please add more as needed
rated_vote_cases: tuple[RatedVoteCase, ...] = (
    RatedVoteCase(
        rated_votes=[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        slate_size=1,
        pareto_efficient_slates=[["s3"]],
        non_extremal_pareto_efficient_slates=[["s3"]],
        expected_assignments=pd.DataFrame(["s3"]*3, columns=["candidate_id"])
    ),
    RatedVoteCase(
        rated_votes=[[4, 2, 3], [4, 2, 3], [4, 2, 3]],
        slate_size=1,
        pareto_efficient_slates=[["s1"]],
        non_extremal_pareto_efficient_slates=[["s1"]],
        expected_assignments=pd.DataFrame(["s1"]*3, columns=["candidate_id"])
    ),
)

# Instances of voting algorithms to test, please add more as needed
voting_algorithms_to_test: Generator[VotingAlgorithm, None, None] = (
    SequentialPhragmenMinimax(),
    SequentialPhragmenMinimax(load_magnitude_method="total"),
)

voting_test_cases: tuple[tuple[str, VotingAlgorithm, RatedVoteCase], ...] = ((algo.name + "___" + rated.name, rated, algo) for rated, algo in itertools.product(rated_vote_cases, voting_algorithms_to_test))

properties_to_evaluate: tuple[str, ...] = (
    "A Basic functionality",
    "B Assignments",
    "C Assignments other columns",
    "01 Pareto efficient",
    "02 Non-extremal Pareto efficient",
)

class AlgorithmEvaluationResult(unittest.TestResult):
    """
    Custom TestResult class to log test results into a DataFrame and write to CSV.
    """
    included_subtests: tuple[str] = properties_to_evaluate[3:]  # Exclude functionality and actual debugging unit tests
    log_filename: Path = get_base_dir_path() / "data" / "voting_algorithm_evals" / f"{get_time_string()}.csv"


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        col_index = pd.MultiIndex.from_product([[case.name for case in rated_vote_cases], self.included_subtests], names=["vote", "subtest"])
        self.results = pd.DataFrame(index=[algo.name for algo in voting_algorithms_to_test], columns=col_index)

    @override
    def addSubTest(self, test, subtest, outcome):
        super().addSubTest(test, subtest, outcome)
        if subtest._message not in self.included_subtests:
            return
        alg_name, vote_name = repr(subtest.test_case).split("___")
        vote_name = vote_name[:-1]
        alg_name = re.sub(r'^.*?_[0-9]+_', '', alg_name)
        subtest_name = subtest._message
        self.results.at[alg_name, (vote_name, subtest_name)] = 1 if outcome is None else 0

    def write_to_csv(self):
        self.results.to_csv(self.log_filename, index=True)


class TestVotingAlgorithms(unittest.TestCase):
    """
    Test the functionality and properties of voting algorithms.
    """
    @parameterized.expand(voting_test_cases)
    def test_voting_algorithm(
        self,
        name: str,
        rated_vote_case: RatedVoteCase,
        voting_algorithm: VotingAlgorithm,
    ):
        """
        Test the sequential Phragmen Minimax algorithm for rated voting.

        # Arguments
        
        """
        slate, assignments = voting_algorithm.vote(
            rated_vote_case.rated_votes,
            rated_vote_case.slate_size,
        )

        with self.subTest(msg=properties_to_evaluate[0]):
            self.assertEqual(len(slate), rated_vote_case.slate_size)
            self.assertEqual(len(set(slate)), len(slate))
            self.assertEqual(len(assignments), len(rated_vote_case.rated_votes))

        # Check that the assignments are valid. For functional debugging only, will be omitted from algorithm evaluation
        if rated_vote_case.expected_assignments is not None:
            with self.subTest(msg=properties_to_evaluate[1]):
                assert pd.DataFrame.equals(assignments.candidate_id, rated_vote_case.expected_assignments.candidate_id)

            with self.subTest(msg=properties_to_evaluate[2]):
                for col in ["utility", "load", "utility_previous", "second_selected_candidate_id"]:
                    if col in rated_vote_case.expected_assignments.columns:
                        assert pd.DataFrame.equals(assignments[col], rated_vote_case.expected_assignments[col])
                    
        with self.subTest(msg=properties_to_evaluate[3]):
            assert frozenset(slate) in frozenset({frozenset(pareto_slate) for pareto_slate in rated_vote_case.pareto_efficient_slates}), "The selected slate is not among the Pareto efficient slates"

        if rated_vote_case.non_extremal_pareto_efficient_slates is not None:
            with self.subTest(msg=properties_to_evaluate[4]):
                assert frozenset(slate) in {frozenset(pareto_slate) for pareto_slate in rated_vote_case.non_extremal_pareto_efficient_slates}, "The selected slate is not among the non-extremal Pareto efficient slates"

        # We'll add more property tests here

    @classmethod
    def tearDownClass(cls):
        """
        Write the test results to a CSV file after all tests have run.
        """
        suite = unittest.TestLoader().loadTestsFromTestCase(cls)
        result = AlgorithmEvaluationResult()
        suite.run(result)
        result.write_to_csv()



# if __name__ == "__main__":
#     rated_votes = pd.DataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3]], columns=[f"s{i}" for i in range(1, 4)])
#     slate_size = 1
#     algo = SequentialPhragmenMinimax()
#     print(algo.vote(rated_votes, slate_size))
