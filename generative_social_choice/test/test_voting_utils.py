from collections.abc import Callable
from pathlib import Path
import unittest
from typing import Optional, Sequence, Generator, Hashable, override
from dataclasses import dataclass
import sys

import itertools
import pandas as pd
import numpy as np
from jaxtyping import Float, Bool
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
class TestParetoDominates(unittest.TestCase):
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


class TestIsParetoEfficient(unittest.TestCase):
    @parameterized.expand([
        (np.array([[1, 2], [2, 1], [3, 0]]), np.array([True, True, True])),
        (np.array([[1, 2], [2, 1], [2, 0]]), np.array([True, True, False])),
        (np.array([[1, 2], [2, 1], [2, 2]]), np.array([False, False, True])),
        (np.array([[1, 2, 3], [2, 1, 2], [2, 2, 1]]), np.array([True, True, True])),
        (np.array([[1, 2, 3], [0, 1, 2], [0, 2, 1]]), np.array([True, False, False])),
        (np.array([[1, 2, 3], [2, 1, 2], [2, 2, 1], [1, 1, 1]]), np.array([True, True, True, False])),

    ])
    def test_is_pareto_efficient(self, utilities: Float[np.ndarray, "slate metric_type"], expected: Bool[np.ndarray, "slate"]):
        assert np.array_equal(is_pareto_efficient(utilities), expected)

class TestParetoEfficientSlates(unittest.TestCase):
    @parameterized.expand([
        (
            pd.DataFrame([[1, 2], [2, 1], [3, 0]]), 
            1, 
            [lambda x: x.sum()], 
            {frozenset({0, 1, 2})}
        ),
        (
            pd.DataFrame([[1, 3], [2, 1], [3, 0]]), 
            1, 
            [lambda x: x.sum()], 
            {frozenset({0})}
        ),
        (
            pd.DataFrame([[1, 2], [2, 1], [3, 0]]), 
            1, 
            [lambda x: x[0], lambda x: x[1]], 
            {frozenset({0, 1, 2})}
        ),
    ])
    def test_pareto_efficient_slates(
        self,
        rated_votes: pd.DataFrame, 
        slate_size: int, 
        positive_metrics: Sequence[Callable[[Float[np.ndarray, "voter_utility"]], float]], 
        expected: set[frozenset[str]]
    ):
        self.assertEqual(pareto_efficient_slates(rated_votes, slate_size, positive_metrics), expected)

