import random
import string
from dataclasses import dataclass
from typing import override, Tuple, List

import pandas as pd

from generative_social_choice.queries.query_interface import Generator, Agent
from generative_social_choice.utils.gpt_wrapper import LLMLog


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