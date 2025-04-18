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

