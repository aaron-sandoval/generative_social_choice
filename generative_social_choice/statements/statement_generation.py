import random
import string
from dataclasses import dataclass
from typing import override, Tuple, List

import pandas as pd

from generative_social_choice.queries.query_interface import Generator,Agent
from generative_social_choice.utils.gpt_wrapper import LLMLog
from generative_social_choice.queries.query_chatbot_personalization import ChatbotPersonalizationGenerator


class SimplePersonalizationAgent(Agent):
    """Simple agent representation which doesn't require connecting to any LLM
    but can't be used to get approvals.
    
    We use this class in statement generation as computing new approvals
    is unnecessary."""

    def __init__(
        self,
        *,
        id: str,
        survey_responses: pd.DataFrame,
        summary: str,
    ):
        self.id = id
        self.survey_responses = survey_responses
        self.summary = summary

    def get_id(self):
        return self.id

    def get_description(self):
        return self.summary

    def get_approval(
        self, statement: str, use_logprobs: bool = True
    ) -> tuple[float, list[LLMLog]]:
        raise NotImplementedError()


class NamedGenerator(Generator):
    """Interface class for generation methods
    
    Almost the same as Generator, but we want to ensure that the arguments passed to init
    can be obtained later for logging purposes."""
    _init_args: dict={}  # Remember init arguments for logging purposes

    @property
    def name(self):
        return self.__class__.__name__ + "(" + ", ".join(f"{key}={value}" for key, value in self._init_args.items()) + ")"


class DummyGenerator(NamedGenerator):
    """Dummy method that returns random strings as new statements.
    
    Use for test purposes only!"""

    def __init__(self, num_statements: int=5, statement_length: int=20):
        self.num_statements = num_statements
        self.statement_length = statement_length
        self._init_args = {"num_statements": num_statements, "statement_length": statement_length}

    def generate(self, agents: List[Agent]) -> Tuple[List[str], List[LLMLog]]:
        """
        Returns random strings of fixed length with letters and whitespace.
        """
        statements = []
        for _ in range(self.num_statements):
            new_statement = ''.join(random.choices(string.ascii_letters + " ", k=self.statement_length))
            statements.append(new_statement)
        return statements, []
    

class NamedChatbotPersonalizationGenerator(ChatbotPersonalizationGenerator, NamedGenerator):
    """Use an LLM to generate a new statement for the given agents.
    
    This class is very similar to queries.query_chatbot_personalization.ChatbotPersonalizationGenerator,
    but has the additional name property for logging"""
    def __init__(
        self,
        *,
        seed: int | None = None,
        gpt_temperature=0.0,
        model: str = "gpt-4o-mini-2024-07-18",
    ):
        init_args = {"seed": seed, "gpt_temperature": gpt_temperature, "model": model}
        super().__init__(**init_args)
        self._init_args = init_args