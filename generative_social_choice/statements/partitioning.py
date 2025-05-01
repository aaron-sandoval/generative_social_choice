import abc
import json
from typing import List, Optional, override
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from generative_social_choice.queries.query_chatbot_personalization import SimplePersonalizationAgent
from generative_social_choice.utils.gpt_wrapper import Embeddings


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


class OpenAIEmbedding(Embedding):
    """
    Class to compute embeddings using OpenAI's embedding models based on summaries of user responses.
    
    This class uses the Embeddings class from gpt_wrapper.py to compute embeddings for user response summaries.
    """
    
    def __init__(self, model: str = "text-embedding-3-small", use_summary: bool = False):
        """
        Initialize the OpenAIEmbedding class.
        
        Args:
            model (str): The OpenAI embedding model to use. Default is "text-embedding-3-small".
                         Other options include "text-embedding-3-large" and "text-embedding-ada-002".
        """
        self.embedder = Embeddings(model=model)
        self.use_summary = use_summary
    
    def _get_user_string(self, agent: SimplePersonalizationAgent) -> str:
        """
        Create a string with user responses that captures their preferences and opinions.
        
        Args:
            agent (SimplePersonalizationAgent): The agent whose responses will be summarized.
            
        Returns:
            str: A text summary of the user's responses.
        """
        # Get all rating statements and their responses
        df = agent.survey_responses
        rating_df = df[df["detailed_question_type"] == "rating statement"]
        
        # Get free-form responses if available
        free_form_df = df[df["detailed_question_type"] == "free form"]
        
        # Create a summary text that includes all the user's responses
        summary_parts = []
        
        # Add rating statements
        if not rating_df.empty:
            for _, row in rating_df.iterrows():
                statement = row["statement"]
                rating = row["choice_numeric"]
                agreement = "strongly disagrees with" if rating == 1 else \
                           "disagrees with" if rating == 2 else \
                           "is neutral about" if rating == 3 else \
                           "agrees with" if rating == 4 else \
                           "strongly agrees with"
                summary_parts.append(f"User {agreement} the statement: '{statement}'.")
        
        # Add free-form responses
        if not free_form_df.empty:
            for _, row in free_form_df.iterrows():
                question = row["statement"]
                response = row["choice"]
                if isinstance(response, str) and response.strip():  # Check if response is not empty
                    summary_parts.append(f"When asked '{question}', user responded: '{response}'.")
        
        # Combine all parts into a single summary
        summary = " ".join(summary_parts)
        
        return summary
    
    @override
    def compute(self, agents: List[SimplePersonalizationAgent]) -> np.ndarray:
        """
        Compute embeddings for the given list of agents using OpenAI's embedding model.
        
        Args:
            agents (List[SimplePersonalizationAgent]): Agents to compute embeddings for.
            
        Returns:
            np.ndarray: Embedding matrix with embeddings of agents[i] in row i.
        """
        # Generate summaries for each agent
        if self.use_summary:
            user_texts = [agent.summary for agent in agents]
        else:
            user_texts = [self._get_user_string(agent) for agent in agents]
        
        # Compute embeddings for all summaries at once
        embeddings_list, _ = self.embedder.embed(user_texts)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings_list)
        
        return embeddings_array


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

    def compute_similarities(self, agent: SimplePersonalizationAgent, other_agents: List[SimplePersonalizationAgent]) -> np.ndarray:
        """
        Compute cosine similarities between one agent and all other agents.
        
        Args:
            agent (SimplePersonalizationAgent): The agent to compute similarities for
            other_agents (List[SimplePersonalizationAgent]): List of agents to compare against
            
        Returns:
            np.ndarray: Array of cosine similarities between the agent and each other agent
        """
        # Get embeddings for the target agent and other agents
        target_embedding = self.compute([agent])
        other_embeddings = self.compute(other_agents)
        
        # Compute cosine similarities
        similarities = cosine_similarity(target_embedding, other_embeddings)
        
        # Return as 1D array
        return similarities.flatten()


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

if __name__ == "__main__":
    from generative_social_choice.utils.helper_functions import get_base_dir_path
    from generative_social_choice.statements.statement_generation import get_simple_agents
    
    # Get all agents
    agents = get_simple_agents()
    
    # Compute and store embeddings using BaselineEmbedding
    embeddings_file = get_base_dir_path() / "data/demo_data/baseline_embeddings.json"
    baseline_embedding = BaselineEmbedding()
    baseline_embedding.precompute(agents=agents, filepath=embeddings_file)
    
    # Load precomputed embeddings
    embedding = PrecomputedEmbedding(filepath=embeddings_file)
    
    # Pick a target agent and compute similarities to all other agents
    target_agent = agents[0]
    other_agents = agents[1:]
    
    similarities = embedding.compute_similarities(target_agent, other_agents)
    
    # Print results
    print(f"Similarities between agent {target_agent.id} and other agents:")
    for agent, similarity in zip(other_agents, similarities):
        print(f"Agent {agent.id}: {similarity:.4f}")
    
    # Print most similar agents
    most_similar_indices = np.argsort(similarities)[-5:]  # Get indices of 5 most similar agents
    print("\nMost similar agents:")
    for idx in reversed(most_similar_indices):  # Print from most to least similar
        print(f"Agent {other_agents[idx].id}: {similarities[idx]:.4f}")