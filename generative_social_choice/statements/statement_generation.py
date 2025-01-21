import abc
import random
import string
from dataclasses import dataclass
from typing import override

import pandas as pd


@dataclass(frozen=True)
class StatementGenerationMethod(abc.ABC):
    """
    Abstract base class for methods to generate additional statements.
    """
    @abc.abstractmethod
    def generate(
        self,
        survey_responses: pd.DataFrame,
        summaries: pd.DataFrame,
        num_statements: int,
    ) -> list[str]:
        """
        Generate new statements for the voters described in the given data.

        # Arguments
        - `survey_responses`: Data of the whole survey in raw format
        - `summaries`: Generated summaries for each user
        - `num_statements: int`: The number of new statements to be generated

        # Returns
        `statements: list[str]`: A list of additional statements
        """
        pass

@dataclass(frozen=True)
class DummyStatementGeneration(StatementGenerationMethod):
    """Dummy method that returns random strings as new statements.
    
    Use for test purposes only!"""
    statement_length: int = 20

    @override
    def generate(
        self,
        survey_responses: pd.DataFrame,
        summaries: pd.DataFrame,
        num_statements: int,
    ) -> list[str]:
        """
        Returns random strings of fixed length with letters and whitespace.
        """
        statements = []
        for _ in range(num_statements):
            new_statement = ''.join(random.choices(string.ascii_letters + " ", k=self.statement_length))
            statements.append(new_statement)
        return statements