import unittest
from typing import Optional, Sequence
# from collections import frozenset

import pandas as pd

from generative_social_choice.slates.voting_algorithms import (
    SequentialPhragmenMinimax,
)

class TestVotingAlgorithms(unittest.TestCase):
    def seq_phragmen_minimax_rated_test(
        self,
        rated_votes: pd.DataFrame,
        slate_size: int,
        pareto_efficient_slates: Sequence[tuple[int]],
        non_extremal_pareto_efficient_slates: Optional[Sequence[tuple[int]]] = None,
        egalitarian_utilitarian: Optional[float] = None,
        expected_assignments: Optional[pd.DataFrame] = None,
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
        
        voting_algorithm = SequentialPhragmenMinimax()
        slate, assignments = voting_algorithm.vote(
            rated_votes,
            slate_size,
        )

        self.assertEqual(len(slate), slate_size)
        self.assertEqual(len(set(slate)), len(slate))
        self.assertEqual(len(assignments), len(rated_votes))

        # Check that the selected slate is among the Pareto efficient slates
        assert frozenset(slate) in frozenset({frozenset(pareto_slate) for pareto_slate in pareto_efficient_slates}), "The selected slate is not among the Pareto efficient slates"

        if non_extremal_pareto_efficient_slates is not None:
            assert frozenset(slate) not in frozenset({frozenset(pareto_slate) for pareto_slate in non_extremal_pareto_efficient_slates}), "The selected slate is an extremal Pareto efficient slate"

        # Check that the assignments are valid
        if expected_assignments is not None:
            assert pd.DataFrame.equals(assignments.candidate_id, expected_assignments.candidate_id)
            if "utility" in expected_assignments.columns:
                assert pd.DataFrame.equals(assignments.utility, expected_assignments.utility)
            if "load" in expected_assignments.columns:
                assert pd.DataFrame.equals(assignments.load, expected_assignments.load)
            if "utility_previous" in expected_assignments.columns:
                assert pd.DataFrame.equals(assignments["utility_previous"], expected_assignments["utility_previous"])
            if "second_selected_candidate_id" in expected_assignments.columns:
                assert pd.DataFrame.equals(assignments["second_selected_candidate_id"], expected_assignments["second_selected_candidate_id"])

    def test_seq_phragmen_slatesize1(self):
        self.seq_phragmen_minimax_rated_test(
            rated_votes=pd.DataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3]], columns=["s1", "s2", "s3"]),
            slate_size=1,
            pareto_efficient_slates=[["s3"]],
            expected_assignments=pd.DataFrame(["s3"]*3, columns=["candidate_id"])
        )

