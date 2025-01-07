import unittest
from generative_social_choice.paper_replication.compute_matching import (
    optimize_monroe_matching,
)
from gurobipy import setParam


class TestMonroe(unittest.TestCase):
    def test_optimize_monroe_matching_replication(self):
        setParam("OutputFlag", False)
        self.assertTrue(
            optimize_monroe_matching([[1, 3], [3, 4], [5, 6], [8, 10]]) == (1, 0, 0, 1)
        )
