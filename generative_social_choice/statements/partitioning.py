from typing import List, override
import abc

import numpy as np
from sklearn.cluster import KMeans

from generative_social_choice.statements.statement_generation import SimplePersonalizationAgent

#TODO Make it possible to store embeddings to avoid having to call LLMs all the time (or wait for kmeans results)
# (Can be done by having separate script to precompute embeddings to a file, and embedding class to read from a file)

# TODO Implement further embedding methods
# User embeddings can now be done based on
# - Their statements ratings (simplest but ignoring free-form texts)
# - Embedding everything with an LLM
# - Embedding the summary of their responses with an LLM or other NLP methods


class Embedding(abc.ABC):
    """Abstract base class for computing embeddings"""

    @abc.abstractmethod
    def compute(self, agents: List[SimplePersonalizationAgent]) -> np.array:
        """
        Compute embeddings for the given list of agents

        # Arguments
        - `agents: List[SimplePersonalizationAgent]`: Agents to compute embeddings for.
          Note that we use SimplePersonalizationAgent rather than Agent to be sure that all survey responses are
          available.

        # Returns
        `embeddings: np.array`: Embedding matrix with embeddings of `agents[i]` in row i (counting from 0)
        """
        pass


class BaselineEmbedding(Embedding):
    """Method to use ratings of the six survey statements as embeddings"""

    @override
    def compute(self, agents: List[SimplePersonalizationAgent]) -> np.array:
        # First get all statements so that we can fix ordering
        statements = agents[0].survey_responses["statement"].dropna().to_list()

        # Compute these embeddings for all agents
        embeddings = []
        for agent in agents:
            df = agent.survey_responses
            df = df[df["detailed_question_type"]=="rating statement"]
            user_ratings = df.set_index("statement")["choice_numeric"].to_dict()
            embeddings.append([user_ratings[statement] for statement in statements])
        return np.array(embeddings)


class Partition(abc.ABC):
    """Abstract base class for partitioning agents"""

    @abc.abstractmethod
    def assign(self, agents: List[SimplePersonalizationAgent], num_partitions: int) -> List[int]:
        """
        Assign the given agents to different partitions.

        # Arguments
        - `agents: List[SimplePersonalizationAgent]`: Agents to partition.
          Note that we use SimplePersonalizationAgent rather than Agent to be sure that all survey responses are
          available.
        - `num_partitions: int`: Number of partitions to create

        # Returns
        `assignment: List[int]`: Assignment of agents to partitions, where `assignment[i]` contains the index
          of the partition `agents[i]` is assigned to.
        """
        pass


class KMeansClustering(Partition):
    
    def __init__(self, embedding_method: Embedding):
        self.embedding_method = embedding_method
    
    @override
    def assign(self, agents: List[SimplePersonalizationAgent], num_partitions: int) -> List[int]:
        embeddings = self.embedding_method.compute(agents=agents)

        kmeans = KMeans(n_clusters=num_partitions)
        kmeans.fit(embeddings)

        return kmeans.labels_