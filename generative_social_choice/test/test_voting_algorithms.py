import unittest
from typing import Optional, Sequence
import pandas as pd

from generative_social_choice.slates.voting_algorithms import (
    _phragmen_update_assignments,
    seq_phragmen_minimax_rated,
)

class TestVotingAlgorithms(unittest.TestCase):
    def seq_phragmen_minimax_rated_test(
        self,
        rated_votes: pd.DataFrame,
        slate_size: int,
        pareto_efficient_slates: Sequence[list[int]],
        egalitarian_utilitarian: Optional[float] = None,
        expected_assignments: Optional[pd.DataFrame] = None,
    ):
        """
        Test the sequential Phragmen Minimax algorithm for rated voting.

        # Arguments
        - `rated_votes: pd.DataFrame`: Utility of each voter (rows) for each candidate (columns)
        - `slate_size: int`: The number of candidates to be selected
        - `pareto_efficient_slates: Sequence[list[int]]`: Slates that are Pareto efficient on the egalitarian-utilitarian trade-off parameter.
          - One test case is that the selected slate is among the Pareto efficient slates.
        - `egalitarian_utilitarian: Optional[float] = None`: The egalitarian-utilitarian trade-off parameter
        """
        slate, assignments = seq_phragmen_minimax_rated(
            rated_votes,
            slate_size,
            egalitarian_utilitarian,
        )

        self.assertEqual(len(slate), slate_size)
        self.assertEqual(len(set(slate)), len(slate))
        self.assertEqual(len(assignments), len(rated_votes))

        # Check that the selected slate is among the Pareto efficient slates
        self.assertSetEqual(set(slate), set(pareto_efficient_slates))

        # Check that the assignments are valid
        if expected_assignments is not None:
            assert pd.DataFrame.equals(assignments.candidate_id, expected_assignments.candidate_id)
            assert pd.DataFrame.equals(assignments.utility, expected_assignments.utility)
            assert pd.DataFrame.equals(assignments.load, expected_assignments.load)
            assert pd.DataFrame.equals(assignments["utility_previous"], expected_assignments["utility_previous"])
            assert pd.DataFrame.equals(assignments["2nd_selected_candidate_id"], expected_assignments["2nd_selected_candidate_id"])

    def test_seq_phragmen_slatesize1(self):
        self.seq_phragmen_minimax_rated_test(
            rated_votes=pd.DataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3]], columns=["s1", "s2", "s3"]),
            slate_size=1,
            pareto_efficient_slates=[["s3"]],
        )

