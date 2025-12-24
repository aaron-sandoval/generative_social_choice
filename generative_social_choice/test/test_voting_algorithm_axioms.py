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

from generative_social_choice.slates.voting_algorithms import RatedVoteCase
from generative_social_choice.slates.voting_utils import voter_max_utilities_from_slate

# Add the project root directory to the system path
sys.path.append(str(Path(__file__).parent.parent.parent))

from generative_social_choice.slates.voting_algorithm_axioms import (
    IndividualParetoAxiom,
    HappiestParetoAxiom,
    CoverageAxiom,
    MinimumAndTotalUtilityParetoAxiom,
    NonRadicalTotalUtilityAxiom,
    VotingAlgorithmAxiom,
    NonRadicalMinUtilityAxiom,
    GeneralizedLorenzAxiom,
)
from generative_social_choice.test.utilities_for_testing import rated_vote_cases

class TestNonRadicalTotalUtilityAxiom(unittest.TestCase):
    @parameterized.expand([
        (
            rated_vote_cases["Ex Alg2.1"],
            [["s2"]], 
            None
        ),
        (
            rated_vote_cases["Ex Alg1.5"],
            [["s2"]], 
            None
        ),
        (
            rated_vote_cases["Ex 2.1"],
            [["s1", "s2"]],
            None
        ),
        (
            rated_vote_cases["Ex 2.2"],
            [["s1", "s2"], ["s2", "s4"]],
            None
        ),            
    ])
    def test_non_radical_total_utility_axiom(self, rated_vote_case: RatedVoteCase, expected_slates: list[list[str]], max_tradeoff: Optional[float] = None):
        axiom = NonRadicalTotalUtilityAxiom(max_tradeoff=max_tradeoff) if max_tradeoff is not None else NonRadicalTotalUtilityAxiom()
        self.assertEqual(axiom.satisfactory_slates(rated_vote_case.rated_votes, rated_vote_case.slate_size), {frozenset(slate) for slate in expected_slates})

        for slate in expected_slates:
            self.assertTrue(axiom.evaluate_assignment(rated_vote_case.rated_votes, rated_vote_case.slate_size, voter_max_utilities_from_slate(rated_vote_case.rated_votes, slate)))

        # Check that all other slates are invalid
        invalid_slates = set(frozenset(slate) for slate in itertools.combinations(rated_vote_case.rated_votes.columns, r=rated_vote_case.slate_size))
        invalid_slates.difference_update({frozenset(slate) for slate in expected_slates})
        for slate in invalid_slates:
            self.assertFalse(axiom.evaluate_assignment(rated_vote_case.rated_votes, rated_vote_case.slate_size, voter_max_utilities_from_slate(rated_vote_case.rated_votes, slate)))

class TestNonRadicalMinUtilityAxiom(unittest.TestCase):
    @parameterized.expand([
        (
            rated_vote_cases["Ex 4.2"],
            [["s1", "s2"]], 
            None
        ),
        (
            rated_vote_cases["Ex Alg1.4"],
            [["s2"]], 
            None
        ),
        (
            rated_vote_cases["Ex 2.1"],
            [["s1", "s2"]],
            None
        ),
        (
            rated_vote_cases["Ex 2.2"],
            [["s1", "s2"], ["s2", "s4"]],
            None
        ),            
    ])
    def test_non_radical_min_utility_axiom(self, rated_vote_case: RatedVoteCase, expected_slates: list[list[str]], max_tradeoff: Optional[float] = None):
        axiom = NonRadicalMinUtilityAxiom(max_tradeoff=max_tradeoff) if max_tradeoff is not None else NonRadicalMinUtilityAxiom()
        self.assertEqual(axiom.satisfactory_slates(rated_vote_case.rated_votes, rated_vote_case.slate_size), {frozenset(slate) for slate in expected_slates})

        for slate in expected_slates:
            self.assertTrue(axiom.evaluate_assignment(rated_vote_case.rated_votes, rated_vote_case.slate_size, voter_max_utilities_from_slate(rated_vote_case.rated_votes, slate)))

        # Check that all other slates are invalid
        invalid_slates = set(frozenset(slate) for slate in itertools.combinations(rated_vote_case.rated_votes.columns, r=rated_vote_case.slate_size))
        invalid_slates.difference_update({frozenset(slate) for slate in expected_slates})
        for slate in invalid_slates:
            self.assertFalse(axiom.evaluate_assignment(rated_vote_case.rated_votes, rated_vote_case.slate_size, voter_max_utilities_from_slate(rated_vote_case.rated_votes, slate)))


class TestGeneralizedLorenzDominates(unittest.TestCase):
    """Test cases for GeneralizedLorenzAxiom.generalized_lorenz_dominates method."""
    
    def test_clear_dominance(self):
        """Test that A clearly dominates B when all cumulative sums are greater."""
        # A: [1, 2, 3] -> sorted: [1, 2, 3] -> cumulative: [1, 3, 6]
        # B: [0, 1, 2] -> sorted: [0, 1, 2] -> cumulative: [0, 1, 3]
        utilities_a = np.array([1, 2, 3])
        utilities_b = np.array([0, 1, 2])
        self.assertTrue(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(utilities_a, utilities_b)
        )
        self.assertFalse(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(utilities_b, utilities_a)
        )
    
    def test_no_dominance_when_some_lower(self):
        """Test that A doesn't dominate B when some cumulative sums are lower."""
        # A: [1, 2, 3] -> sorted: [1, 2, 3] -> cumulative: [1, 3, 6]
        # B: [2, 1, 4] -> sorted: [1, 2, 4] -> cumulative: [1, 3, 7]
        # B has higher third cumulative, so A doesn't dominate
        utilities_a = np.array([1, 2, 3])
        utilities_b = np.array([2, 1, 4])
        self.assertTrue(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(utilities_b, utilities_a)
        )
        self.assertFalse(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(utilities_a, utilities_b)
        )
    
    def test_equal_curves_no_dominance(self):
        """Test that equal curves don't dominate each other."""
        utilities_a = np.array([1, 2, 3])
        utilities_b = np.array([1, 2, 3])
        self.assertFalse(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(utilities_a, utilities_b)
        )
        self.assertFalse(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(utilities_b, utilities_a)
        )
    
    def test_different_lengths(self):
        """Test that different length arrays raise ValueError."""
        utilities_a = np.array([1, 2, 3])
        utilities_b = np.array([1, 2])
        with self.assertRaises(ValueError) as context:
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(utilities_a, utilities_b)
        self.assertIn("same length", str(context.exception))
        self.assertIn("len(utilities_a)=3", str(context.exception))
        self.assertIn("len(utilities_b)=2", str(context.exception))
        
        with self.assertRaises(ValueError):
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(utilities_b, utilities_a)
    
    def test_tolerance_rel_tol_only(self):
        """Test tolerance handling with rel_tol only."""
        # Create utilities that are very close but not equal
        utilities_a = np.array([1.0, 2.0, 3.0])
        utilities_b = np.array([1.0 + 1e-10, 2.0 + 1e-10, 3.0 + 1e-10])
        # With default rel_tol=1e-9, these should be considered equal
        self.assertFalse(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(
                utilities_b, utilities_a, rel_tol=1e-9, abs_tol=0.0
            )
        )
        
        # With tighter tolerance, should detect dominance
        utilities_a = np.array([1.0, 2.0, 3.0])
        utilities_b = np.array([1.0, 2.0, 2.9])
        self.assertTrue(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(
                utilities_a, utilities_b, rel_tol=1e-9, abs_tol=0.0
            )
        )
    
    def test_tolerance_abs_tol_only(self):
        """Test tolerance handling with abs_tol only."""
        # Create utilities that differ by small absolute amount
        utilities_a = np.array([1.0, 2.0, 3.0])
        utilities_b = np.array([1.0, 2.0, 3.0 - 1e-6])
        # With abs_tol=1e-5, these should be considered equal
        self.assertFalse(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(
                utilities_a, utilities_b, rel_tol=0.0, abs_tol=1e-5
            )
        )
        
        # With smaller abs_tol, should detect dominance
        self.assertTrue(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(
                utilities_a, utilities_b, rel_tol=0.0, abs_tol=1e-7
            )
        )
    
    def test_tolerance_both_rel_and_abs(self):
        """Test tolerance handling with both rel_tol and abs_tol."""
        # Create utilities where rel_tol would catch it but abs_tol wouldn't
        utilities_a = np.array([1.0, 2.0, 3.0])
        utilities_b = np.array([1.0, 2.0, 3.0 - 0.01])
        # With rel_tol=0.1 and abs_tol=0.001, the difference (0.01) is within rel_tol
        # but not within abs_tol. Since we use max(rel_tol * max_abs, abs_tol),
        # rel_tol * 3.0 = 0.3 > 0.01, so they're considered close
        self.assertFalse(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(
                utilities_a, utilities_b, rel_tol=0.1, abs_tol=0.001
            )
        )
        
        # With tighter rel_tol, should detect dominance
        self.assertTrue(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(
                utilities_a, utilities_b, rel_tol=0.001, abs_tol=0.001
            )
        )

    def test_partial_dominance(self):
        """Test case where A is >= B in all positions but not strictly greater."""
        # A: [1, 2, 3] -> sorted: [1, 2, 3] -> cumulative: [1, 3, 6]
        # B: [1, 2, 3] -> sorted: [1, 2, 3] -> cumulative: [1, 3, 6]
        # Equal, so no dominance
        utilities_a = np.array([1, 2, 3])
        utilities_b = np.array([1, 2, 3])
        self.assertFalse(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(utilities_a, utilities_b)
        )
        
        # A: [1, 2, 4] -> sorted: [1, 2, 4] -> cumulative: [1, 3, 7]
        # B: [1, 2, 3] -> sorted: [1, 2, 3] -> cumulative: [1, 3, 6]
        # A >= B in all, and strictly greater in last, so A dominates
        utilities_a = np.array([1, 2, 4])
        utilities_b = np.array([1, 2, 3])
        self.assertTrue(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(utilities_a, utilities_b)
        )
    
    def test_negative_utilities(self):
        """Test with negative utilities."""
        utilities_a = np.array([-1, 0, 1])
        utilities_b = np.array([-2, -1, 0])
        # A: sorted: [-1, 0, 1] -> cumulative: [-1, -1, 0]
        # B: sorted: [-2, -1, 0] -> cumulative: [-2, -3, -3]
        # A dominates B
        self.assertTrue(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(utilities_a, utilities_b)
        )
    
    def test_mixed_positive_negative(self):
        """Test with mixed positive and negative utilities."""
        utilities_a = np.array([-1, 1, 2])
        utilities_b = np.array([-2, 0, 1])
        # A: sorted: [-1, 1, 2] -> cumulative: [-1, 0, 2]
        # B: sorted: [-2, 0, 1] -> cumulative: [-2, -2, -1]
        # A dominates B
        self.assertTrue(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(utilities_a, utilities_b)
        )
    
    def test_single_element(self):
        """Test with single element arrays."""
        utilities_a = np.array([2.0])
        utilities_b = np.array([1.0])
        self.assertTrue(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(utilities_a, utilities_b)
        )
        self.assertFalse(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(utilities_b, utilities_a)
        )
        
        # Equal single elements
        utilities_a = np.array([1.0])
        utilities_b = np.array([1.0])
        self.assertFalse(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(utilities_a, utilities_b)
        )
    
    def test_large_arrays(self):
        """Test with larger arrays."""
        utilities_a = np.array([1, 2, 3, 4, 5])
        utilities_b = np.array([1, 2, 3, 4, 4])
        # A dominates B (last cumulative sum is greater)
        self.assertTrue(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(utilities_a, utilities_b)
        )
        
        utilities_a = np.array([1, 2, 3, 4, 4])
        utilities_b = np.array([1, 2, 3, 4, 5])
        # B dominates A
        self.assertFalse(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(utilities_a, utilities_b)
        )
    
    def test_reordered_utilities(self):
        """Test that order doesn't matter (sorting happens internally)."""
        # These should produce the same generalized Lorenz curves
        utilities_a = np.array([3, 1, 2])
        utilities_b = np.array([1, 2, 3])
        # Both sorted: [1, 2, 3] -> cumulative: [1, 3, 6]
        # So equal, no dominance
        self.assertFalse(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(utilities_a, utilities_b)
        )
        
        # Now with different values
        utilities_a = np.array([4, 1, 2])
        utilities_b = np.array([1, 2, 3])
        # A: sorted: [1, 2, 4] -> cumulative: [1, 3, 7]
        # B: sorted: [1, 2, 3] -> cumulative: [1, 3, 6]
        # A dominates B
        self.assertTrue(
            GeneralizedLorenzAxiom.generalized_lorenz_dominates(utilities_a, utilities_b)
        )
    
    @parameterized.expand([
        (1e-12, 0.0),  # Very small rel_tol, no abs_tol
        (1e-12, 1e-12),  # Both very small
        (1e-6, 0.0),  # Small rel_tol, no abs_tol
        (1e-6, 1e-6),  # Both small
        (1e-3, 0.0),  # Medium rel_tol, no abs_tol
        (1e-3, 1e-3),  # Both medium
    ])
    def test_tolerance_combinations(self, rel_tol: float, abs_tol: float):
        """Test various combinations of rel_tol and abs_tol."""
        # Create utilities where A should dominate B
        utilities_a = np.array([1.0, 2.0, 3.001])
        utilities_b = np.array([1.0, 2.0, 3.0])
        # A: cumulative: [1, 3, 6.001], B: cumulative: [1, 3, 6]
        # A should dominate if tolerance is small enough
        
        result = GeneralizedLorenzAxiom.generalized_lorenz_dominates(
            utilities_a, utilities_b, rel_tol=rel_tol, abs_tol=abs_tol
        )
        
        # If tolerance is small, A dominates (result = True)
        # Small tolerance, A should dominate
        self.assertTrue(result,
            f"With rel_tol={rel_tol}, abs_tol={abs_tol}, A should dominate B")


class TestGeneralizedLorenzAxiom(unittest.TestCase):
    """Test cases for GeneralizedLorenzAxiom class."""
    
    def test_evaluate_assignment_basic(self):
        """Test evaluate_assignment with a simple case."""
        axiom = GeneralizedLorenzAxiom()
        
        # Create a simple rated votes matrix
        rated_votes = pd.DataFrame({
            's1': [1, 2, 3],
            's2': [0, 1, 2],
            's3': [2, 3, 4]
        })
        
        # Assignment using s3 (best utilities)
        assignments = pd.DataFrame({
            'candidate_id': ['s3', 's3', 's3']  # All get s3 (highest utility)
        })
        
        # s3 gives [2, 3, 4], s1 gives [1, 2, 3], s2 gives [0, 1, 2]
        # s3's cumulative: [2, 5, 9], s1's cumulative: [1, 3, 6], s2's cumulative: [0, 1, 3]
        # s3 dominates both s1 and s2, so s3 assignment should satisfy the axiom
        result = axiom.evaluate_assignment(rated_votes, slate_size=1, assignments=assignments)
        self.assertTrue(result)
    
    def test_evaluate_assignment_violation(self):
        """Test evaluate_assignment when axiom is violated."""
        axiom = GeneralizedLorenzAxiom()
        
        # Create rated votes where one slate clearly dominates
        rated_votes = pd.DataFrame({
            's1': [1, 1, 1],
            's2': [2, 2, 2],
        })
        
        # Assignment using s1 (dominated by s2)
        assignments = pd.DataFrame({
            'candidate_id': ['s1', 's1', 's1']
        })
        
        # s2 dominates s1, so this should violate the axiom
        result = axiom.evaluate_assignment(rated_votes, slate_size=1, assignments=assignments)
        self.assertFalse(result)
    
    @parameterized.expand([
        (rated_vote_cases["Ex 2.1"],),
        (rated_vote_cases["Ex 2.2"],),
        (rated_vote_cases["Ex 4.2"],),  # Bad egalitarian tradeoff - good for testing
        (rated_vote_cases["Ex 4.1"],),
        (rated_vote_cases["Ex 1.1"],),
        (rated_vote_cases["Ex 1.1 modified"],),
    ])
    def test_satisfactory_slates_with_test_cases(self, rated_vote_case: RatedVoteCase):
        """Test satisfactory_slates with existing test cases."""
        axiom = GeneralizedLorenzAxiom()
        satisfactory = axiom.satisfactory_slates(
            rated_vote_case.rated_votes, 
            
            rated_vote_case.slate_size
        )
        
        # Should return a set of frozensets
        self.assertIsInstance(satisfactory, set)
        self.assertGreater(len(satisfactory), 0, "Should have at least one satisfactory slate")
        for slate in satisfactory:
            self.assertIsInstance(slate, frozenset)
        
        # All satisfactory slates should satisfy evaluate_assignment
        for slate in satisfactory:
            assignments = voter_max_utilities_from_slate(
                rated_vote_case.rated_votes, list(slate)
            )
            self.assertTrue(
                axiom.evaluate_assignment(
                    rated_vote_case.rated_votes,
                    rated_vote_case.slate_size,
                    assignments
                ),
                f"Slate {slate} should satisfy evaluate_assignment"
            )
