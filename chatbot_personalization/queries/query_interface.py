from abc import ABC, abstractmethod
from typing import List, Tuple
from chatbot_personalization.utils.gpt_wrapper import LLMLog


class Agent(ABC):
    @abstractmethod
    def get_id(self) -> str:
        pass

    @abstractmethod
    def get_approval(self, statement: str) -> Tuple[float, List[LLMLog]]:
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Return description of agent for use in generative queries (e.g. summary of user survey responses).
        """
        pass


class Generator(ABC):
    @abstractmethod
    def generate(self, agents: List[Agent]) -> Tuple[List[str], List[LLMLog]]:
        pass
