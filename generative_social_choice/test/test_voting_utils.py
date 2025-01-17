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
    pareto_efficient_slates
)

class TestVotingUtils(unittest.TestCase):
    def test_voter_utilities(self):
        self.assertEqual(voter_utilities(pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}), pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})), [1, 2, 3])
