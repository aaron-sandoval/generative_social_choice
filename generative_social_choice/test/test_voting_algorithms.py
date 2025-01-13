import unittest
from typing import Optional, Sequence, Generator, Hashable
from dataclasses import dataclass
import itertools
import pandas as pd
from parameterized import parameterized

from generative_social_choice.slates.voting_algorithms import (
    SequentialPhragmenMinimax,
    VotingAlgorithm,
)


@dataclass
class RatedVoteCase:
    """
    A voting case with rated votes and expected results.

    # Arguments
    - `rated_votes: pd.DataFrame | list[list[int | float]]`: Utility of each voter (rows) for each candidate (columns)
      - If passed as a nested list, it's converted to a DataFrame with columns named `s1`, `s2`, etc.
    - `slate_size: int`: The number of candidates to be selected
    - `pareto_efficient_slates: Sequence[list[int]]`: Slates that are Pareto efficient on the egalitarian-utilitarian trade-off parameter.
    - `non_extremal_pareto_efficient_slates: Optional[Sequence[list[int]]] = None`: Slates that are non-extremal Pareto efficient on the egalitarian-utilitarian trade-off parameter.
        - Subset of `pareto_efficient_slates` which don't make arbitrarily large egalitarian-utilitarian sacrifices in either direction.
    - `egalitarian_utilitarian: Optional[float] = None`: The egalitarian-utilitarian trade-off parameter
    - `expected_assignments: Optional[pd.DataFrame] = None`: A singular expected assignment with the following columns:
        - `candidate_id`: The candidate to which the voter is assigned
        - Other columns not guaranteed to always be present, should always be checked in the unit tests
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
            cols_str = " ".join(str(col) + ": " + ",".join(str(x) for x in self.rated_votes[col]) 
                              for col in self.rated_votes.columns)
            self.name = f"k={self.slate_size}; {cols_str}"
        elif self.name[:2] == "k=":
            self.name = f"k={self.slate_size}; {self.name}"


rated_vote_cases: tuple[RatedVoteCase, ...] = (
    RatedVoteCase(
        rated_votes=[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        slate_size=1,
        pareto_efficient_slates=[["s3"]],
        expected_assignments=pd.DataFrame(["s3"]*3, columns=["candidate_id"])
    ),
    RatedVoteCase(
        rated_votes=[[4, 2, 3], [4, 2, 3], [4, 2, 3]],
        slate_size=1,
        pareto_efficient_slates=[["s1"]],
        expected_assignments=pd.DataFrame(["s1"]*3, columns=["candidate_id"])
    ),
)
voting_algorithms_to_test: Generator[VotingAlgorithm, None, None] = (
    SequentialPhragmenMinimax(),
    SequentialPhragmenMinimax(load_magnitude_method="total"),
)
voting_test_cases: tuple[tuple[str, VotingAlgorithm, RatedVoteCase], ...] = ((algo.name + "_" + rated.name, rated, algo) for rated, algo in itertools.product(rated_vote_cases, voting_algorithms_to_test))


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

        with self.subTest(msg="1 Functionality"):
            self.assertEqual(len(slate), rated_vote_case.slate_size)
            self.assertEqual(len(set(slate)), len(slate))
            self.assertEqual(len(assignments), len(rated_vote_case.rated_votes))

        with self.subTest(msg="2 Pareto efficient"):
            assert frozenset(slate) in frozenset({frozenset(pareto_slate) for pareto_slate in rated_vote_case.pareto_efficient_slates}), "The selected slate is not among the Pareto efficient slates"

        if rated_vote_case.non_extremal_pareto_efficient_slates is not None:
            with self.subTest(msg="3 Non-extremal Pareto efficient"):
                assert frozenset(slate) not in {frozenset(pareto_slate) for pareto_slate in rated_vote_case.non_extremal_pareto_efficient_slates}, "The selected slate is an extremal Pareto efficient slate"

        # Check that the assignments are valid
        if rated_vote_case.expected_assignments is not None:
            with self.subTest(msg="4 Assignments"):
                assert pd.DataFrame.equals(assignments.candidate_id, rated_vote_case.expected_assignments.candidate_id)
            with self.subTest(msg="5 Assignments other columns"):
                if "utility" in rated_vote_case.expected_assignments.columns:
                    assert pd.DataFrame.equals(assignments.utility, rated_vote_case.expected_assignments.utility)
                if "load" in rated_vote_case.expected_assignments.columns:
                    assert pd.DataFrame.equals(assignments.load, rated_vote_case.expected_assignments.load)
                if "utility_previous" in rated_vote_case.expected_assignments.columns:
                    assert pd.DataFrame.equals(assignments["utility_previous"], rated_vote_case.expected_assignments["utility_previous"])
                if "second_selected_candidate_id" in rated_vote_case.expected_assignments.columns:
                    assert pd.DataFrame.equals(assignments["second_selected_candidate_id"], rated_vote_case.expected_assignments["second_selected_candidate_id"])
