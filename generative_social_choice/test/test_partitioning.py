import sys
import unittest
from pathlib import Path
from typing import List

import numpy as np
from parameterized import parameterized

# Add the project root directory to the system path
sys.path.append(str(Path(__file__).parent.parent.parent))

from generative_social_choice.queries.query_interface import Agent
from generative_social_choice.statements.statement_generation import get_simple_agents
from generative_social_choice.statements.partitioning import (
    BaselineEmbedding,
    KMeansClustering,
)

agents = get_simple_agents()

test_cases = [

]

class TestBaselineEmbedding(unittest.TestCase):
    @parameterized.expand([
        (agents[:10],),
        (agents[10:18],),
        (agents[30:48],),
        (agents,),
    ])
    def test_compute(self, agents: List[Agent]):
        embedding_method = BaselineEmbedding()
        embeddings = embedding_method.compute(agents=agents)
        assert type(embeddings) is np.ndarray, f"Type should be numpy array but is {type(embeddings)}"
        assert embeddings.shape[0]==len(agents)

        pair_embeddings = embedding_method.compute(agents=[agents[0], agents[0]])
        assert type(pair_embeddings) is np.ndarray, f"Type should be numpy array but is {type(pair_embeddings)}"
        assert pair_embeddings.shape[0]==2
        assert sum(abs(pair_embeddings[0]-pair_embeddings[1]))<1e-9


class TestKMeans(unittest.TestCase):
    @parameterized.expand([
        (agents[:10], 3),
        (agents[10:23], 4),
        (agents[30:35], 2),
    ])
    def test_compute(self, agents: List[Agent], num_partitions: int):
        partitioning = KMeansClustering(num_partitions=num_partitions, embedding_method=BaselineEmbedding())
        assignments = partitioning.assign(agents=agents)

        assert len(assignments)==len(agents), "Each agent has to be assigned to a partition!"
        assert len(set(assignments))<=num_partitions

        for value in assignments:
            assert value in range(num_partitions)