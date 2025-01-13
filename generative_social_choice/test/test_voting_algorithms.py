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


@dataclass(frozen=True)
class RatedVoteCase:
    rated_votes: pd.DataFrame | list[list[int | float]]
    slate_size: int
    pareto_efficient_slates: set[frozenset[Hashable]]
    non_extremal_pareto_efficient_slates: Optional[set[frozenset[Hashable]]] = None
    expected_assignments: Optional[pd.DataFrame] = None
    name: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.rated_votes, list):
            self.rated_votes = pd.DataFrame(self.rated_votes, columns=[f"s{i}" for i in range(len(self.rated_votes))])

        if self.name is None:
            cols_str = "_".join(str(col) + ": " + ",".join(str(x) for x in self.rated_votes[col]) 
                              for col in self.rated_votes.columns)
            self.name = f"k={self.slate_size}; {cols_str}"
        elif self.name[:2] == "k=":
            self.name = f"k={self.slate_size}; {self.name}"


rated_vote_cases: tuple[RatedVoteCase, ...] = (
    RatedVoteCase(
        rated_votes=pd.DataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3]], columns=["s1", "s2", "s3"]),
        slate_size=1,
        pareto_efficient_slates=[["s3"]],
        expected_assignments=pd.DataFrame(["s3"]*3, columns=["candidate_id"])
    ),
)
voting_algorithms_to_test: Generator[VotingAlgorithm, None, None] = (
    SequentialPhragmenMinimax(),
)
voting_test_cases: tuple[tuple[VotingAlgorithm, RatedVoteCase], ...] = ((rated.name, rated, algo) for rated, algo in itertools.product(voting_algorithms_to_test, rated_vote_cases))


class TestVotingAlgorithms(unittest.TestCase):
    @parameterized.expand(voting_test_cases)
    def voting_algorithm_test(
        self,
        name: str,
        voting_algorithm: VotingAlgorithm,
        rated_vote_case: RatedVoteCase,
    ):
        """
        Test the sequential Phragmen Minimax algorithm for rated voting.

        # Arguments
        - `rated_votes: pd.DataFrame`: Utility of each voter (rows) for each candidate (columns)
        - `slate_size: int`: The number of candidates to be selected
        - `pareto_efficient_slates: Sequence[list[int]]`: Slates that are Pareto efficient on the egalitarian-utilitarian trade-off parameter.
        - `non_extremal_pareto_efficient_slates: Optional[Sequence[list[int]]] = None`: Slates that are non-extremal Pareto efficient on the egalitarian-utilitarian trade-off parameter.
          - Subset of `pareto_efficient_slates` which don't make arbitrarily large egalitarian-utilitarian sacrifices in either direction.
        - `egalitarian_utilitarian: Optional[float] = None`: The egalitarian-utilitarian trade-off parameter
        - `expected_assignments: Optional[pd.DataFrame] = None`: If there is a singular expected assignment w
        """
        
        slate, assignments = voting_algorithm.vote(
            rated_vote_case.rated_votes,
            rated_vote_case.slate_size,
        )

        with self.subTest(msg="Functionality"):
            self.assertEqual(len(slate), rated_vote_case.slate_size)
            self.assertEqual(len(set(slate)), len(slate))
            self.assertEqual(len(assignments), len(rated_vote_case.rated_votes))

        with self.subTest(msg="Pareto efficient"):
            assert frozenset(slate) in frozenset({frozenset(pareto_slate) for pareto_slate in rated_vote_case.pareto_efficient_slates}), "The selected slate is not among the Pareto efficient slates"

        if rated_vote_case.non_extremal_pareto_efficient_slates is not None:
            with self.subTest(msg="Non-extremal Pareto efficient"):
                assert frozenset(slate) not in {frozenset(pareto_slate) for pareto_slate in rated_vote_case.non_extremal_pareto_efficient_slates}, "The selected slate is an extremal Pareto efficient slate"

        # Check that the assignments are valid
        if rated_vote_case.expected_assignments is not None:
            with self.subTest(msg="Assignments"):
                assert pd.DataFrame.equals(assignments.candidate_id, rated_vote_case.expected_assignments.candidate_id)
            with self.subTest(msg="Assignments other columns"):
                if "utility" in rated_vote_case.expected_assignments.columns:
                    assert pd.DataFrame.equals(assignments.utility, rated_vote_case.expected_assignments.utility)
                if "load" in rated_vote_case.expected_assignments.columns:
                    assert pd.DataFrame.equals(assignments.load, rated_vote_case.expected_assignments.load)
                if "utility_previous" in rated_vote_case.expected_assignments.columns:
                    assert pd.DataFrame.equals(assignments["utility_previous"], rated_vote_case.expected_assignments["utility_previous"])
                if "second_selected_candidate_id" in rated_vote_case.expected_assignments.columns:
                    assert pd.DataFrame.equals(assignments["second_selected_candidate_id"], rated_vote_case.expected_assignments["second_selected_candidate_id"])

    # def test_seq_phragmen_slatesize1(self):
    #     self.voting_algorithm_test(
    #         voting_algorithm=SequentialPhragmenMinimax(),
    #         rated_votes=pd.DataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3]], columns=["s1", "s2", "s3"]),
    #         slate_size=1,
    #         pareto_efficient_slates=[["s3"]],
    #         expected_assignments=pd.DataFrame(["s3"]*3, columns=["candidate_id"])
    #     )

