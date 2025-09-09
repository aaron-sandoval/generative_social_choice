"""
Test file to verify that noise augmentation in RatedVoteCase objects produces unique noise
patterns across different augmented cases, ensuring proper randomization while maintaining
repeatability.
"""

import unittest
from pathlib import Path
import sys
from typing import List
import itertools

import pandas as pd
import numpy as np
from parameterized import parameterized

# Add the project root directory to the system path
sys.path.append(str(Path(__file__).parent.parent.parent))

from generative_social_choice.test.utilities_for_testing import rated_vote_cases
from generative_social_choice.slates.voting_algorithms import RatedVoteCase


class TestNoiseUniqueness(unittest.TestCase):
    """Test that noise augmentation produces unique patterns across augmented cases."""
    
    def setUp(self):
        """Set up test with the first 3 rated vote cases."""
        self.test_cases = list(rated_vote_cases.values())[:3]
        self.tolerance = 1e-7  # Tolerance for floating point comparisons
    
    def test_first_three_cases_exist(self):
        """Verify that we have at least 3 test cases to work with."""
        self.assertGreaterEqual(len(self.test_cases), 3, 
                               "Need at least 3 test cases for noise uniqueness testing")
    
    @parameterized.expand([
        (0, "Simple 1"),
        (1, "Simple 2"), 
        (2, "Simple 3"),
    ])
    def test_augmented_cases_have_unique_noise(self, case_index: int, case_name: str):
        """Test that each augmented case has unique noise patterns."""
        case = self.test_cases[case_index]
        self.assertEqual(case.name, case_name, f"Expected case {case_index} to be '{case_name}'")
        
        # Get augmented cases
        augmented_cases = case.augmented_cases
        
        # Should have more than just the original case if noise augmentation is enabled
        if case.noise_augmentation:
            self.assertGreater(len(augmented_cases), 1, 
                             f"Case '{case_name}' should have augmented cases with noise")
        
        # Compare each pair of augmented cases to ensure they're different
        for i, j in itertools.combinations(range(len(augmented_cases)), 2):
            case_i = augmented_cases[i]
            case_j = augmented_cases[j]
            
            # Cases should have the same shape
            self.assertEqual(case_i.shape, case_j.shape, 
                           f"Augmented cases {i} and {j} should have the same shape")
            
            # Calculate the absolute difference between the two cases
            diff = np.abs(case_i.values - case_j.values)
            max_diff = np.max(diff)
            
            # The cases should be different (max difference should be above tolerance)
            self.assertGreater(max_diff, self.tolerance,
                             f"Augmented cases {i} and {j} in '{case_name}' are too similar "
                             f"(max difference: {max_diff:.2e}). Expected unique noise patterns.")
    
    def test_noise_consistency_across_initializations(self):
        """Test that the same case produces identical noise across multiple initializations."""
        for case_index in range(3):
            original_case = self.test_cases[case_index]
            
            # Create a new instance with the same parameters
            new_case = RatedVoteCase(
                name=original_case.name,
                rated_votes=original_case.rated_votes.values.tolist(),
                slate_size=original_case.slate_size,
                noise_augmentation=original_case.noise_augmentation,
                noise_augmentation_methods=original_case.noise_augmentation_methods
            )
            
            # Get augmented cases from both instances
            original_augmented = original_case.augmented_cases
            new_augmented = new_case.augmented_cases
            
            # Should have the same number of augmented cases
            self.assertEqual(len(original_augmented), len(new_augmented),
                           f"Case '{original_case.name}' should produce the same number "
                           f"of augmented cases across initializations")
            
            # Each corresponding augmented case should be identical
            for i, (orig, new) in enumerate(zip(original_augmented, new_augmented)):
                # Use pandas equals for exact comparison
                self.assertTrue(orig.equals(new),
                              f"Augmented case {i} in '{original_case.name}' should be "
                              f"identical across initializations")
    
    def test_different_cases_have_different_noise(self):
        """Test that different cases produce different noise patterns."""
        # Compare the first augmented case from each of the first 3 test cases
        if len(self.test_cases) >= 3:
            aug_case_0 = self.test_cases[0].augmented_cases
            aug_case_1 = self.test_cases[1].augmented_cases
            aug_case_2 = self.test_cases[2].augmented_cases
            
            # Ensure all have at least one augmented case
            if len(aug_case_0) > 0 and len(aug_case_1) > 0 and len(aug_case_2) > 0:
                # Compare first augmented case from each
                cases_to_compare = [
                    (0, 1, aug_case_0[0], aug_case_1[0]),
                    (0, 2, aug_case_0[0], aug_case_2[0]),
                    (1, 2, aug_case_1[0], aug_case_2[0])
                ]
                
                for i, j, case_i, case_j in cases_to_compare:
                    # Only compare if they have the same shape
                    if case_i.shape == case_j.shape:
                        diff = np.abs(case_i.values - case_j.values)
                        max_diff = np.max(diff)
                        
                        self.assertGreater(max_diff, self.tolerance,
                                         f"Different test cases {i} and {j} should produce "
                                         f"different noise patterns (max difference: {max_diff:.2e})")
    
    def test_noise_magnitude_within_expected_range(self):
        """Test that the noise added is within the expected magnitude range."""
        for case_index in range(3):
            case = self.test_cases[case_index]
            original_votes = case.rated_votes
            augmented_cases = case.augmented_cases
            
            if case.noise_augmentation and len(augmented_cases) > 0:
                for i, augmented in enumerate(augmented_cases):
                    # Calculate the noise (difference from original)
                    noise = augmented.values - original_votes.values
                    
                    # Check that noise magnitude is reasonable
                    # Based on the default noise augmentation methods, noise should be small
                    max_noise_magnitude = np.max(np.abs(noise))
                    
                    # Should be non-zero (unless this is the original case)
                    if not np.allclose(noise, 0, atol=self.tolerance):
                        self.assertGreater(max_noise_magnitude, self.tolerance,
                                         f"Noise in augmented case {i} of '{case.name}' "
                                         f"should be non-zero")
                    
                    # Should be within reasonable bounds (less than 1.0 for typical cases)
                    self.assertLess(max_noise_magnitude, 1.0,
                                  f"Noise magnitude in augmented case {i} of '{case.name}' "
                                  f"seems too large: {max_noise_magnitude}")
    
    def test_augmented_cases_preserve_dataframe_structure(self):
        """Test that augmented cases preserve the original DataFrame structure."""
        for case_index in range(3):
            case = self.test_cases[case_index]
            original_votes = case.rated_votes
            augmented_cases = case.augmented_cases
            
            for i, augmented in enumerate(augmented_cases):
                # Should be a DataFrame
                self.assertIsInstance(augmented, pd.DataFrame,
                                    f"Augmented case {i} should be a DataFrame")
                
                # Should have the same shape
                self.assertEqual(augmented.shape, original_votes.shape,
                               f"Augmented case {i} should have the same shape as original")
                
                # Should have the same index and columns
                pd.testing.assert_index_equal(augmented.index, original_votes.index,
                                            f"Augmented case {i} should have the same index")
                pd.testing.assert_index_equal(augmented.columns, original_votes.columns,
                                            f"Augmented case {i} should have the same columns")
    
    def test_explicit_seed_produces_consistent_results(self):
        """Test that explicit seed setting produces consistent results."""
        # Create two identical cases with explicit seeds
        test_votes = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        
        case1 = RatedVoteCase(
            name="explicit_seed_test",
            rated_votes=test_votes,
            slate_size=2,
            noise_augmentation=True,
            base_seed=12345
        )
        
        case2 = RatedVoteCase(
            name="explicit_seed_test", 
            rated_votes=test_votes,
            slate_size=2,
            noise_augmentation=True,
            base_seed=12345
        )
        
        aug1 = case1.augmented_cases
        aug2 = case2.augmented_cases
        
        self.assertEqual(len(aug1), len(aug2),
                        "Cases with same seed should have same number of augmented cases")
        
        for i, (a1, a2) in enumerate(zip(aug1, aug2)):
            self.assertTrue(a1.equals(a2),
                          f"Augmented case {i} should be identical with same explicit seed")


if __name__ == '__main__':
    unittest.main()
