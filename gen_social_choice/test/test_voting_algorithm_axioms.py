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
            1, 
            [["s1"]], 
            None
        ),
        (
            rated_vote_cases["Ex Alg1.5"], 
            1, 
            [["s1"], ["s2"]], 
            None
        ),
    ])
    def test_non_radical_total_utility_axiom(self, rated_votes: pd.DataFrame, slate_size: int, expected_slates: list[list[str]], max_tradeoff: Optional[float] = None):
        axiom = NonRadicalTotalUtilityAxiom(max_tradeoff=max_tradeoff) if max_tradeoff is not None else NonRadicalTotalUtilityAxiom()
        self.assertEqual(axiom.satisfactory_slates(rated_votes, slate_size), {frozenset(slate) for slate in expected_slates})
