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

# Add the project root directory to the system path
sys.path.append(str(Path(__file__).parent.parent.parent))

from generative_social_choice.slates.voting_algorithms import (
    SequentialPhragmenMinimax,
    GreedyTotalUtilityMaximization,
    ExactTotalUtilityMaximization,
    LPTotalUtilityMaximization,
    VotingAlgorithm,
)
from generative_social_choice.utils.helper_functions import get_time_string, get_base_dir_path
from generative_social_choice.slates.voting_utils import voter_utilities, mth_highest_utility

@dataclass
class RatedVoteCase:
    """
    A voting case with rated votes and sets of possible results which satisfy various properties.

    # Arguments
    - `rated_votes: pd.DataFrame | list[list[int | float]]`: Utility of each voter (rows) for each candidate (columns)
      - If passed as a nested list, it's converted to a DataFrame with columns named `s1`, `s2`, etc.
    - `slate_size: int`: The number of candidates to be selected
    - `pareto_efficient_slates: Optional[Sequence[list[int]]] = None`: Slates that are Pareto efficient on the egalitarian-utilitarian trade-off parameter.
      - Egalitarian objective: Maximize the minimum utility among all individual voters
      - Utilitarian objective: Maximize the total utility among all individual voters
    - `non_extremal_pareto_efficient_slates: Optional[Sequence[list[int]]] = None`: Slates that are non-extremal Pareto efficient on the egalitarian-utilitarian trade-off parameter.
        - Subset of `pareto_efficient_slates` which don't make arbitrarily large egalitarian-utilitarian sacrifices in either direction.
        - Ex: For Example Alg2.1, s1 is Pareto efficient, but not non-extremal Pareto efficient because it makes an arbitrarily large egalitarian sacrifice for an incremental utilitarian gain.
    - `expected_assignments: Optional[pd.DataFrame] = None`: An expected assignment of voters to candidates with the following columns:
        - `candidate_id`: The candidate to which the voter is assigned
        - Other columns not guaranteed to always be present, used for functional testing only. They should always be checked in the unit tests
    """
    rated_votes: pd.DataFrame | list[list[int | float]]
    slate_size: int
    pareto_efficient_slates: Optional[set[frozenset[str]]] = None
    non_extremal_pareto_efficient_slates: Optional[set[frozenset[str]]] = None
    # expected_assignments: Optional[pd.DataFrame] = None
    name: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.rated_votes, list):
            self.rated_votes = pd.DataFrame(self.rated_votes, columns=[f"s{i}" for i in range(1, len(self.rated_votes[0]) + 1)])

        if self.name is None:
            cols_str = "_".join(str(col) + "_" + "_".join(str(x).replace(".", "p") for x in self.rated_votes[col]) 
                              for col in self.rated_votes.columns)
            self.name = f"k_{self.slate_size}_{cols_str}"
        elif self.name is not None:
            # Format name to be compatible as a Python function name
            self.name = self.name.replace('.', 'p')
            self.name = re.sub(r'[^a-zA-Z0-9_]', '_', self.name)
            self.name = re.sub(r'^[^a-zA-Z_]+', '', self.name)  # Remove leading non-letters
            # self.name = re.sub(r'_+', '_', self.name)  # Collapse multiple underscores
            # self.name = self.name.strip('_')  # Remove trailing underscores


# The voting cases to test, please add more as needed
rated_vote_cases: tuple[RatedVoteCase, ...] = (
    RatedVoteCase(
        rated_votes=[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        slate_size=1,
        pareto_efficient_slates=[["s3"]],
        non_extremal_pareto_efficient_slates=[["s3"]],
        # expected_assignments=pd.DataFrame(["s3"]*3, columns=["candidate_id"])
    ),
    RatedVoteCase(
        rated_votes=[[4, 2, 3], [4, 2, 3], [4, 2, 3]],
        slate_size=1,
        pareto_efficient_slates=[["s1"]],
        non_extremal_pareto_efficient_slates=[["s1"]],
        # expected_assignments=pd.DataFrame(["s1"]*3, columns=["candidate_id"])
    ),
    RatedVoteCase(
        rated_votes=[[1, 1] , [1.1, 1], [1, 1]],
        slate_size=1,
        pareto_efficient_slates=[["s1"]],
        non_extremal_pareto_efficient_slates=[["s1"]],
        # expected_assignments=pd.DataFrame(["s1"]*3, columns=["candidate_id"])
    ),
    RatedVoteCase(
        name="Ex 1.1",
        rated_votes=[
            [3, 2, 0, 0],
            [0, 2, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 3, 2],
            [0, 0, 0, 2],
            [0, 0, 0, 2],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex 1.1 modified",
        rated_votes=[
            [6, 2, 0, 0],
            [0, 2, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 6, 2],
            [0, 0, 0, 2],
            [0, 0, 0, 2],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex A.1",
        rated_votes=[
            [2, 0, 1, 1],
            [2, 2, 1, 0],
            [0, 2, 1, 0],
            [0, 0, 0, 2],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex 1.3",
        rated_votes=[
            [2, 0, 0, 0],
            [2, 2, 1, 0],
            [0, 2, 1, 1],
            [0, 0, 1, 2],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex 2.1",
        rated_votes=[
            [2, 0, 0],
            [2, 2, 1],
            [0, 2, 0],
            [0, 0, 1],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex 2.2",
        rated_votes=[
            [5, 0, 0, 1],
            [0, 5, 0, 1],
            [0, 5, 2, 1],
            [0, 0, 1, 1],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex 3.1",
        rated_votes=[
            [2, 0, 1, 0, 0, 0],
            [1, 2, 0, 0, 0, 0],
            [0, 1, 2, 0, 0, 0],
            [0, 0, 0, 2, 0, 1],
            [0, 0, 0, 1, 2, 0],
            [0, 0, 0, 0, 1, 2],
        ],
        slate_size=3,
    ),
    RatedVoteCase(
        name="Ex 4.1",
        rated_votes=[
            [2, 0, 0, 1],
            [0, 5, 0, 1],
            [0, 5, 2, 1],
            [0, 0, 1, 1],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex 4.4",
        rated_votes=[
            [3, 2, 0, 0, 0, 1],
            [0, 2, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 1],
            [0, 0, 3, 2, 0, 0],
            [0, 0, 0, 2, 3, 0],
            [0, 0, 0, 2, 0, 1],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex B.1",
        rated_votes=[
            [2, 0, 0],
            [2, 0, 2],
            [0, 2, 0],
            [0, 2, 1],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex B.2",
        rated_votes=[
            [9, 0, 0, 1],
            [9, 0, 1, 0],
            [0, 9, 0, 1],
            [0, 9, 1, 0],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex B.3",
        rated_votes=[
            [3, 0, 2],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex C.1",
        rated_votes=[
            [1, 0, 3],
            [1, 0, 1],
            [0, 2, 0],
            [0, 2, 0],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex C.2",
        rated_votes=[
            [3, 0, 0, 2],
            [3, 0, 2, 0],
            [0, 1, 2, 0],
            [0, 1, 0, 2],
        ],
        slate_size=2,
    ),
    RatedVoteCase(
        name="Ex D.1",
        rated_votes=[
            [4, 0, 0, 0, 0],
            [4, 3, 0, 0, 0],
            [4, 3, 2, 0, 0],
            [0, 3, 2, 1, 0],
            [0, 0, 2, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
        ],
        slate_size=3,
    ),
    RatedVoteCase(
        name="Ex Alg1.3",
        rated_votes=[
            [1, 1],
            [1, 1],
            [1, 4],
        ],
        slate_size=1,
    ),
    RatedVoteCase(
        name="Ex Alg1.5",
        rated_votes=[
            [1, 1],
            [1, 1],
            [1, 2],
        ],
        slate_size=1,
    ),
    RatedVoteCase(
        name="Ex Alg A.1",
        rated_votes=[
            [1.01, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ],
        slate_size=3,
    ),
)

# Instances of voting algorithms to test, please add more as needed
# voting_algorithms_to_test: Generator[VotingAlgorithm, None, None] = all_instances(VotingAlgorithm)
voting_algorithms_to_test = (
    GreedyTotalUtilityMaximization(),
    ExactTotalUtilityMaximization(),
    LPTotalUtilityMaximization(),
    SequentialPhragmenMinimax(),
    SequentialPhragmenMinimax(load_magnitude_method="total"),
)

voting_test_cases: tuple[tuple[str, VotingAlgorithm, RatedVoteCase], ...] = tuple((algo.name + "___" + rated.name, rated, algo) for rated, algo in itertools.product(rated_vote_cases, voting_algorithms_to_test))

axioms_to_evaluate: tuple[str, ...] = (
    "00 (Minimum, total utility) Pareto efficient",
    "01 (Minimum, total utility) Non-extremal Pareto efficient",
    "02 Individual Pareto efficient",
    "03 m-th happiest person Pareto efficient",
    "04 Maximum coverage",
)

class AlgorithmEvaluationResult(unittest.TestResult):
    """
    Custom TestResult class to log test results into a DataFrame and write to CSV.
    """
    included_subtests: set[str] = set(axioms_to_evaluate)
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


class TestVotingAlgorithms(unittest.TestCase):
    """
    Test the functionality and properties of voting algorithms.
    """
    @parameterized.expand(voting_test_cases)
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

        # Check that the assignments are valid. For functional debugging only, will be omitted from algorithm evaluation
        if rated_vote_case.expected_assignments is not None:
            with self.subTest(msg="B Assignments"):
                assert pd.DataFrame.equals(assignments.candidate_id, rated_vote_case.expected_assignments.candidate_id)

            with self.subTest(msg="C Assignments other columns"):
                for col in ["utility", "load", "utility_previous", "second_selected_candidate_id"]:
                    if col in rated_vote_case.expected_assignments.columns:
                        assert pd.DataFrame.equals(assignments[col], rated_vote_case.expected_assignments[col])

    # We'll add more property tests here

    @parameterized.expand(voting_test_cases)
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

        # TODO: make this a function in voting_utils
        if rated_vote_case.pareto_efficient_slates is not None:
            with self.subTest(msg=axioms_to_evaluate[0]):
                assert frozenset(W) in frozenset({frozenset(pareto_slate) for pareto_slate in rated_vote_case.pareto_efficient_slates}), "The selected slate is not among the Pareto efficient slates"

        # TODO: make this a function in voting_utils
        if rated_vote_case.non_extremal_pareto_efficient_slates is not None:
            with self.subTest(msg=axioms_to_evaluate[1]):
                assert frozenset(W) in {frozenset(pareto_slate) for pareto_slate in rated_vote_case.non_extremal_pareto_efficient_slates}, "The selected slate is not among the non-extremal Pareto efficient slates"

        # Get utilities for computed solution
        w_utilities = np.array(voter_utilities(rated_vote_case.rated_votes, assignments))

        # Initialize flags for each check
        individual_pareto_flag = True
        mth_happiest_flag = True
        maximum_coverage_flag = True

        # Check if there is any strictly better slate
        for Wprime in itertools.combinations(rated_vote_case.rated_votes.columns, r=rated_vote_case.slate_size):
            if all([individual_pareto_flag, mth_happiest_flag, maximum_coverage_flag]):
                break
            
            # Skip if same slate
            if sorted(list(Wprime)) == sorted(list(W)):
                continue

            # Compute utilities (using optimal assignment for given slate)
            wprime_utilities = rated_vote_case.rated_votes.loc[:, Wprime].max(axis=1).to_numpy()

            # 1st check: There is no slate for which total utility strictly improves and for no member the utility decreases
            if individual_pareto_flag:
                if wprime_utilities.sum() > w_utilities.sum() and (wprime_utilities >= w_utilities).all():
                    individual_pareto_flag = False

            # 2nd check: No other slate has m-th happiest person at least as good for all m and strictly better for at least one m’
            # (Note that we get the m-th happiest person function by sorting the utilities in descending order.)
            if mth_happiest_flag:
                mth_happiest = np.sort(w_utilities)[::-1]
                mth_happiest_prime = np.sort(wprime_utilities)[::-1]
                if (mth_happiest_prime > mth_happiest).any() and (mth_happiest_prime <= mth_happiest).all():
                    mth_happiest_flag = False

            # 3rd check (representing as many people as possible):
            # There is no other slate with at least the same total utility and a threshold m,
            # such that m'-th happiest person for that slate is >= for all m'>=m and > for some m*
            if maximum_coverage_flag:
                matching_total_utility = wprime_utilities.sum() >= w_utilities.sum()
                strictly_greater_ms = np.where(wprime_utilities > w_utilities)[0]
                if len(strictly_greater_ms) > 0:
                    # If from this index on, m-th happiest person never has lower utility in w_prime, then the threshold is valid
                    threshold_exists = (wprime_utilities[strictly_greater_ms.max():] < w_utilities[strictly_greater_ms.max():]).sum() == 0
                    if matching_total_utility and threshold_exists:
                        maximum_coverage_flag = False

        # Assertions after the loop
        with self.subTest(msg=axioms_to_evaluate[2]):
            assert individual_pareto_flag, "There is a slate with strictly greater total utility and no lesser utility for any individual member"

        with self.subTest(msg=axioms_to_evaluate[3]):
            assert mth_happiest_flag, "There is a slate with a strictly better m-th happiest person curve"

        with self.subTest(msg=axioms_to_evaluate[4]):
            assert maximum_coverage_flag, "There is a slate which represents more people"


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestVotingAlgorithms)
    runner = unittest.TextTestRunner(resultclass=AlgorithmEvaluationResult)
    runner.run(suite)
