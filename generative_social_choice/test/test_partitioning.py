import sys
import unittest
from pathlib import Path
from typing import List

import numpy as np
from parameterized import parameterized

# Add the project root directory to the system path
sys.path.append(str(Path(__file__).parent.parent.parent))

from generative_social_choice.utils.helper_functions import get_base_dir_path
from generative_social_choice.queries.query_interface import Agent
from generative_social_choice.statements.statement_generation import get_simple_agents
from generative_social_choice.statements.partitioning import (
    BaselineEmbedding,
    KMeansClustering,
    PrecomputedEmbedding,
    PrecomputedPartition,
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

    @parameterized.expand([
        (agents[:10],),
        (agents[10:18],),
        (agents[30:48],),
        (agents,),
    ])
    def test_precompute(self, agents: List[Agent]):
        filepath = get_base_dir_path() / "test/test_data/temp.json"
        embedding_method = BaselineEmbedding()
        embedding_method.precompute(agents=agents, filepath=filepath)

        # Now read these embeddings and verify that they contain what we expect
        pre_embedding = PrecomputedEmbedding(filepath=filepath)
        assert len(pre_embedding.agent_ids)==len(agents)
        for agent in agents:
            assert agent.id in pre_embedding.agent_ids

        # We should get the same embeddings if we call the method directly
        embeddings = embedding_method.compute(agents=agents)
        precomputed_embeddings = pre_embedding.compute(agents=agents)
        assert np.abs(embeddings - precomputed_embeddings).mean()<1e-8

        # It should also work to consider subsets of agents
        embeddings = embedding_method.compute(agents=agents[:2])
        precomputed_embeddings = pre_embedding.compute(agents=agents[:2])
        assert np.abs(embeddings - precomputed_embeddings).mean()<1e-8


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

    @parameterized.expand([
        (agents[:10], 3),
        (agents[10:23], 4),
        (agents[30:35], 2),
    ])
    def test_precompute(self, agents: List[Agent], num_partitions: int):
        filepath = get_base_dir_path() / "test/test_data/temp.json"
        partitioning = KMeansClustering(num_partitions=num_partitions, embedding_method=BaselineEmbedding())
        partitioning.precompute(agents=agents, filepath=filepath)

        # Now read these embeddings and verify that they contain what we expect
        pre_partition = PrecomputedPartition(filepath=filepath)
        assert len(pre_partition.agent_ids)==len(agents)
        for agent in agents:
            assert agent.id in pre_partition.agent_ids
        assert len(set(pre_partition.assignments))<=num_partitions