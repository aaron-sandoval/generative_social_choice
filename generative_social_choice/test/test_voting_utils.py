from collections.abc import Callable
from pathlib import Path
import unittest
from typing import Sequence
import sys

import pandas as pd
import numpy as np
from jaxtyping import Float, Bool
from parameterized import parameterized

# Add the project root directory to the system path
sys.path.append(str(Path(__file__).parent.parent.parent))

from generative_social_choice.slates.voting_utils import (
    is_pareto_efficient,
    pareto_efficient_slates,
    pareto_dominates,
    filter_candidates_by_individual_pareto_efficiency
)
class TestParetoDominates(unittest.TestCase):
    @parameterized.expand([
        ((1, 2, 3), (2, 1, 3), False),
        ((1, 2, 3), (1, 2, 3), False),
        ((1, 2, 3), (1, 3, 2), False),
        ((1, 2, 3), (1, 1, 1), True),
        ((1., 2., 3.), (1., 2., 3.), False),
        ((1,), (0,), True),
        ((), (), False),
        ((1,), (1, 2), ValueError),
        (1, 2, TypeError),
    ])
    def test_pareto_dominates(self, a: tuple[float, ...], b: tuple[float, ...], expected):
        if isinstance(expected, bool):
            self.assertEqual(pareto_dominates(a, b), expected)
        else:
            with self.assertRaises(expected):
                pareto_dominates(a, b)


class TestIsParetoEfficient(unittest.TestCase):
    @parameterized.expand([
        (np.array([[1, 2], [2, 1], [3, 0]]), np.array([True, True, True])),
        (np.array([[1, 2], [2, 1], [2, 0]]), np.array([True, True, False])),
        (np.array([[1, 2], [2, 1], [2, 2]]), np.array([False, False, True])),
        (np.array([[4, 2], [5, 2], [4, 1]]), np.array([False, True, False])),
        (np.array([[1, 2, 3], [2, 1, 2], [2, 2, 1]]), np.array([True, True, True])),
        (np.array([[1, 2, 3], [0, 1, 2], [0, 2, 1]]), np.array([True, False, False])),
        (np.array([[1, 2, 3], [2, 1, 2], [2, 2, 1], [1, 1, 1]]), np.array([True, True, True, False])),
        (np.array([[1., 2.], [1., 2.]]), np.array([True, True])),

    ])
    def test_is_pareto_efficient(self, utilities: Float[np.ndarray, "slate metric_type"], expected: Bool[np.ndarray, "slate"]):  # noqa: F821
        assert np.array_equal(is_pareto_efficient(utilities), expected)

class TestParetoEfficientSlates(unittest.TestCase):
    @parameterized.expand([
        (
            pd.DataFrame([[1, 2, 3], 
                          [2, 1, 0]]), 
            1, 
            [lambda x: x.sum()], 
            {frozenset(slate) for slate in [[0], [1], [2]]}
        ),
        (
            pd.DataFrame([[1, 2, 3], 
                          [3, 1, 0]]), 
            1, 
            [lambda x: x.sum()], 
            {frozenset({0})}
        ),
        (
            pd.DataFrame([[1, 2, 3], 
                          [2, 1, 0]]), 
            1, 
            [lambda x: x[0], lambda x: x[1]], 
            {frozenset(slate) for slate in [[0], [1], [2]]}
        ),
        (
            pd.DataFrame([[1, 2, 3], 
                          [2, 1, 0]]), 
            2, 
            [lambda x: x.sum(), lambda x: x.min()], 
            {frozenset(slate) for slate in [[0, 2]]}
        ),
        (
            pd.DataFrame([[1, 2, 3], 
                          [2, 1, 0]]), 
            2, 
            [lambda x: x.min()], 
            {frozenset(slate) for slate in [[0, 1], [0, 2]]}
        ),
        (
            pd.DataFrame([[1, 2, 3], 
                          [2, 1, 0]]), 
            2, 
            [lambda x: x.max()], 
            {frozenset(slate) for slate in [[0, 2], [1, 2]]}
        ),
        (
            pd.DataFrame([[1, 2, 3], 
                          [2, 1, 0]]), 
            2, 
            [lambda x: -x.std()], 
            {frozenset(slate) for slate in [[0, 1]]}
        ),
        (
            pd.DataFrame([[1, 2, 3], 
                          [2, 3, 4]]), 
            2, 
            [lambda x: -x.std()], 
            {frozenset(slate) for slate in [[0, 1], [1, 2], [0, 2]]}
        ),
    ])
    def test_pareto_efficient_slates(
        self,
        rated_votes: pd.DataFrame, 
        slate_size: int, 
        positive_metrics: Sequence[Callable[[Float[np.ndarray, "voter_utility"]], float]],  # noqa: F821
        expected: set[frozenset[str]]
    ):
        self.assertEqual(pareto_efficient_slates(rated_votes, slate_size, positive_metrics), expected)


class TestFilterCandidatesByIndividualParetoEfficiency(unittest.TestCase):
    """Test cases for filter_candidates_by_individual_pareto_efficiency function."""
    
    def test_all_candidates_efficient(self):
        """Test case where all candidates are individually Pareto efficient."""
        # Each candidate is best for at least one voter
        rated_votes = pd.DataFrame([
            [3, 1, 2],  # Voter 0 prefers candidate 0
            [1, 3, 2],  # Voter 1 prefers candidate 1  
            [1, 2, 3]   # Voter 2 prefers candidate 2
        ], columns=['A', 'B', 'C'])
        
        result = filter_candidates_by_individual_pareto_efficiency(rated_votes)
        
        # All candidates should be kept since each is best for someone
        pd.testing.assert_frame_equal(result, rated_votes)
    
    def test_some_candidates_inefficient(self):
        """Test case where some candidates are individually Pareto inefficient."""
        # Candidate B is dominated by candidate A for all voters
        rated_votes = pd.DataFrame([
            [3, 1, 2],  # A > C > B
            [3, 1, 2],  # A > C > B
            [3, 1, 2]   # A > C > B
        ], columns=['A', 'B', 'C'])
        
        result = filter_candidates_by_individual_pareto_efficiency(rated_votes)
        
        # Only A and C should remain (B is dominated)
        expected = rated_votes[['A']]
        pd.testing.assert_frame_equal(result, expected)
    
    def test_single_candidate(self):
        """Test case with only one candidate."""
        rated_votes = pd.DataFrame([
            [2],
            [3],
            [1]
        ], columns=['A'])
        
        result = filter_candidates_by_individual_pareto_efficiency(rated_votes)
        
        # Single candidate should always be kept
        pd.testing.assert_frame_equal(result, rated_votes)
    
    def test_single_voter(self):
        """Test case with only one voter."""
        rated_votes = pd.DataFrame([
            [3, 1, 2, 4]
        ], columns=['A', 'B', 'C', 'D'])
        
        result = filter_candidates_by_individual_pareto_efficiency(rated_votes)
        
        # All candidates should be kept for single voter case
        pd.testing.assert_frame_equal(result, rated_votes[['D']])
    
    def test_identical_utilities(self):
        """Test case where some candidates have identical utilities."""
        rated_votes = pd.DataFrame([
            [2, 2, 3],  # A and B are tied
            [1, 1, 2],  # A and B are tied
            [3, 3, 1]   # A and B are tied
        ], columns=['A', 'B', 'C'])
        
        result = filter_candidates_by_individual_pareto_efficiency(rated_votes)
        
        # All candidates should be kept since A and B are equivalent
        pd.testing.assert_frame_equal(result, rated_votes)
    
    def test_column_order_preservation(self):
        """Test that column order is preserved in the output."""
        rated_votes = pd.DataFrame([
            [1, 3, 2, 1],  # B is best
            [2, 1, 3, 1],  # C is best
            [3, 1, 1, 2]   # A is best
        ], columns=['A', 'B', 'C', 'D'])
        
        result = filter_candidates_by_individual_pareto_efficiency(rated_votes)
        
        # D should be filtered out (dominated by others)
        expected = rated_votes[['A', 'B', 'C']]
        pd.testing.assert_frame_equal(result, expected)
        
        # Check column order is preserved
        self.assertEqual(list(result.columns), ['A', 'B', 'C'])
    
    def test_float_utilities(self):
        """Test with floating point utilities."""
        rated_votes = pd.DataFrame([
            [2.5, 1.2, 3.7],
            [1.8, 2.9, 0.5],
            [3.1, 0.8, 2.2]
        ], columns=['X', 'Y', 'Z'])
        
        result = filter_candidates_by_individual_pareto_efficiency(rated_votes)
        
        # All candidates should be efficient in this case
        pd.testing.assert_frame_equal(result, rated_votes)
    
    def test_mixed_data_types(self):
        """Test with mixed integer and float utilities."""
        rated_votes = pd.DataFrame([
            [2, 1.5, 3],
            [1.5, 3, 2],
            [3, 2, 1.5]
        ], columns=['P', 'Q', 'R'])
        
        result = filter_candidates_by_individual_pareto_efficiency(rated_votes)
        
        # All candidates should be efficient
        pd.testing.assert_frame_equal(result, rated_votes)
    
    def test_clearly_dominated_candidate(self):
        """Test case with a clearly dominated candidate."""
        rated_votes = pd.DataFrame([
            [5, 1, 4, 3],  # A > C > D > B
            [4, 1, 5, 3],  # C > A > D > B  
            [3, 1, 4, 5]   # D > C > A > B
        ], columns=['A', 'B', 'C', 'D'])
        
        result = filter_candidates_by_individual_pareto_efficiency(rated_votes)
        
        # B should be filtered out as it's worst for all voters
        expected = rated_votes[['A', 'C', 'D']]
        pd.testing.assert_frame_equal(result, expected)
    
    def test_consistent_results(self):
        """Test that the function gives consistent results when called multiple times."""
        rated_votes = pd.DataFrame([
            [3, 1, 2],
            [1, 3, 2], 
            [2, 1, 3]
        ], columns=['A', 'B', 'C'])
        
        # Test multiple calls - should give same result
        result1 = filter_candidates_by_individual_pareto_efficiency(rated_votes)
        result2 = filter_candidates_by_individual_pareto_efficiency(rated_votes)
        result3 = filter_candidates_by_individual_pareto_efficiency(rated_votes)
        
        pd.testing.assert_frame_equal(result1, result2)
        pd.testing.assert_frame_equal(result2, result3)
        pd.testing.assert_frame_equal(result1, rated_votes)  # All should be efficient
    
    def test_empty_dataframe_columns(self):
        """Test behavior with DataFrame that has no columns."""
        rated_votes = pd.DataFrame(index=[0, 1, 2])
        
        result = filter_candidates_by_individual_pareto_efficiency(rated_votes)
        
        # Should return empty DataFrame with same index
        expected = pd.DataFrame(index=[0, 1, 2])
        pd.testing.assert_frame_equal(result, expected)
    
    def test_zero_utilities(self):
        """Test with zero and negative utilities."""
        rated_votes = pd.DataFrame([
            [0, -1, 1],
            [-1, 0, 1],
            [1, -1, 0]
        ], columns=['A', 'B', 'C'])
        
        result = filter_candidates_by_individual_pareto_efficiency(rated_votes)
        
        pd.testing.assert_frame_equal(result, rated_votes[['A', 'C']])
    
    def test_large_utility_differences(self):
        """Test with large differences in utility values."""
        rated_votes = pd.DataFrame([
            [1000, 1, 500],
            [1, 1000, 500],
            [500, 1, 1000]
        ], columns=['A', 'B', 'C'])
        
        result = filter_candidates_by_individual_pareto_efficiency(rated_votes)
        
        # All should be efficient
        pd.testing.assert_frame_equal(result, rated_votes)
    
    @parameterized.expand([
        # Test case 1: Two equivalent candidates, one dominated
        (
            pd.DataFrame([
                [3, 3, 1],  # A and B tied, C worst
                [2, 2, 1],  # A and B tied, C worst
                [1, 1, 0]   # A and B tied, C worst
            ], columns=['A', 'B', 'C']),
            ['A', 'B']  # C should be filtered out
        ),
        # Test case 2: All different preferences
        (
            pd.DataFrame([
                [1, 2, 3, 4],
                [4, 3, 2, 1],
                [2, 4, 1, 3],
                [3, 1, 4, 2]
            ], columns=['W', 'X', 'Y', 'Z']),
            ['W', 'X', 'Y', 'Z']  # All should remain
        ),
        # Test case 3: One candidate dominates all others
        (
            pd.DataFrame([
                [5, 1, 2, 3],
                [5, 2, 1, 3],
                [5, 3, 1, 2]
            ], columns=['BEST', 'BAD1', 'BAD2', 'BAD3']),
            ['BEST']  # Only BEST should remain
        )
    ])
    def test_parameterized_cases(self, rated_votes: pd.DataFrame, expected_columns: list[str]):
        """Parameterized test cases for various scenarios."""
        result = filter_candidates_by_individual_pareto_efficiency(rated_votes)
        
        # Check that the right columns are present
        self.assertEqual(sorted(result.columns.tolist()), sorted(expected_columns))
        
        # Check that the data is preserved for kept columns
        expected = rated_votes[expected_columns]
        pd.testing.assert_frame_equal(result, expected)

