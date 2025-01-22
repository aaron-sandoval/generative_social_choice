from typing import List, override
import abc

import numpy as np

from generative_social_choice.statements.statement_generation import SimplePersonalizationAgent

# TODO Implement further embedding methods
# User embeddings can now be done based on
# - Their statements ratings (simplest but ignoring free-form texts)
# - Embedding everything with an LLM
# - Embedding the summary of their responses with an LLM or other NLP methods

#TODO! Proceed with using the embeddings for clustering
#TODO! Use clustering in Generator method

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
