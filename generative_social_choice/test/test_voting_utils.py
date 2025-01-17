from pathlib import Path
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

from generative_social_choice.slates.voting_utils import (
    voter_utilities,
    mth_highest_utility,
    is_pareto_efficient,
    pareto_efficient_slates,
    pareto_dominates
)
class TestVotingUtils(unittest.TestCase):
    @parameterized.expand([
        ((1, 2, 3), (2, 1, 3), False),
        ((1, 2, 3), (1, 2, 3), False),
        ((1, 2, 3), (1, 3, 2), False),
        ((1, 2, 3), (1, 1, 1), True),
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

    # def test_is_pareto_efficient(self):
