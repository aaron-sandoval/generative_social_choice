import abc
import json
from typing import List, override, Optional
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from gen_social_choice.queries.query_chatbot_personalization import SimplePersonalizationAgent


# TODO Implement further embedding methods
# User embeddings can now be done based on
# - Embedding everything with an LLM (ideally using embedding endpoint)
# - Embedding the summary of their responses with an LLM or other NLP methods


##################################################
# Embedding methods
##################################################

def store_embeddings(filepath: Path, embeddings: np.ndarray, agent_ids: List[str]):
    """
    Saves embeddings and corresponding agent IDs to a file.

    Args:
        filepath (str): Path to the file where data will be saved.
        embeddings (numpy.ndarray): A 2D array where each row is the embedding for an agent.
        agent_ids (list of str): A list of agent IDs corresponding to the embeddings.
    """
    data = {
        "agent_ids": agent_ids,
        "embeddings": embeddings.tolist()
    }
    with open(filepath, "w") as file:
        json.dump(data, file)


class Embedding(abc.ABC):
    """Abstract base class for computing embeddings"""

    @abc.abstractmethod
    def compute(self, agents: List[SimplePersonalizationAgent]) -> np.ndarray:
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
    
    def precompute(self, agents: List[SimplePersonalizationAgent], filepath: Path):
        """
        Compute embeddings and store them to a file

        # Arguments
        - `agents: List[SimplePersonalizationAgent]`: Agents to compute embeddings for.
          Note that we use SimplePersonalizationAgent rather than Agent to be sure that all survey responses are
          available.
        - `filepath: Path`: File to store the embeddings.
        """
        embeddings = self.compute(agents=agents)
        store_embeddings(filepath=filepath, agent_ids=[agent.id for agent in agents], embeddings=embeddings)


class BaselineEmbedding(Embedding):
    """Method to use ratings of the six survey statements as embeddings"""

    @override
    def compute(self, agents: List[SimplePersonalizationAgent]) -> np.ndarray:
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
    

class PrecomputedEmbedding(Embedding):
    """Class to read precomputed embeddings from a file"""

    def __init__(self, filepath: Path):
        with open(filepath, "r") as file:
          data = json.load(file)

        self.agent_ids = data["agent_ids"]
        self.embeddings = np.array(data["embeddings"])
        assert self.embeddings.shape[0]==len(self.agent_ids)

    @override
    def compute(self, agents: List[SimplePersonalizationAgent]) -> np.ndarray:
        requested_ids = [agent.id for agent in agents]
        for i in requested_ids:
            assert i in self.agent_ids

        # Map requested IDs to indices
        id_to_index = {agent_id: idx for idx, agent_id in enumerate(self.agent_ids)}

        # Filter embeddings for requested IDs
        filtered_ids = [agent_id for agent_id in requested_ids if agent_id in id_to_index]
        filtered_embeddings = np.array([self.embeddings[id_to_index[agent_id]] for agent_id in filtered_ids])

        return filtered_embeddings


##################################################
# Partitioning
##################################################

def store_assignments(filepath: Path, assignments: np.ndarray, agent_ids: List[str]):
    """
    Saves partitioning and corresponding agent IDs to a file.

    Args:
        filepath (str): Path to the file where data will be saved.
        assignments (numpy.ndarray): A 1D array where each entry is the index of the assign partition for an agent.
        agent_ids (list of str): A list of agent IDs corresponding to the partitioning.
    """
    data = {
        "agent_ids": agent_ids,
        "assignments": assignments.tolist()
    }
    with open(filepath, "w") as file:
        json.dump(data, file)


class Partition(abc.ABC):
    """Abstract base class for partitioning agents"""
    num_partitions: int

    @abc.abstractmethod
    def assign(self, agents: List[SimplePersonalizationAgent]) -> List[int]:
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
    
    def precompute(self, agents: List[SimplePersonalizationAgent], filepath: Path):
        """
        Compute assignments and store them to a file

        # Arguments
        - `agents: List[SimplePersonalizationAgent]`: Agents to compute embeddings for.
          Note that we use SimplePersonalizationAgent rather than Agent to be sure that all survey responses are
          available.
        - `filepath: Path`: File to store the assignments.
        """
        assignments = self.assign(agents=agents)
        store_assignments(filepath=filepath, agent_ids=[agent.id for agent in agents], assignments=assignments)


class KMeansClustering(Partition):
    
    def __init__(self, num_partitions: int, embedding_method: Embedding, seed: Optional[int]=None):
        self.num_partitions = num_partitions
        self.embedding_method = embedding_method
        self.seed = seed
    
    @override
    def assign(self, agents: List[SimplePersonalizationAgent]) -> List[int] | np.ndarray:
        embeddings = self.embedding_method.compute(agents=agents)

        kmeans = KMeans(n_clusters=self.num_partitions, random_state=self.seed)
        kmeans.fit(embeddings)

        return kmeans.labels_
    

class PrecomputedPartition(Partition):
    """Class to read precomputed assignments from a file"""

    def __init__(self, filepath: Path):
        with open(filepath, "r") as file:
          data = json.load(file)

        self.agent_ids = data["agent_ids"]
        self.assignments = np.array(data["assignments"])
        assert self.assignments.shape[0]==len(self.agent_ids)

    @override
    def assign(self, agents: List[SimplePersonalizationAgent]) -> List[int] | np.ndarray:
        requested_ids = [agent.id for agent in agents]
        for i in requested_ids:
            assert i in self.agent_ids

        # Map requested IDs to indices
        id_to_index = {agent_id: idx for idx, agent_id in enumerate(self.agent_ids)}

        # Filter embeddings for requested IDs
        filtered_ids = [agent_id for agent_id in requested_ids if agent_id in id_to_index]
        filtered_assignments = np.array([self.assignments[id_to_index[agent_id]] for agent_id in filtered_ids])

        return filtered_assignments