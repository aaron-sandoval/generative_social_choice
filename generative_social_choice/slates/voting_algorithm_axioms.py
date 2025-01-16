import abc
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class VotingAlgorithmAxiom(abc.ABC):
    """
    An axiom which a voting algorithm may satisfy.

    An algorithm satisfies an axiom if it satisfies the axiom for all possible rated votes and slate sizes.
    """
    name: str

    @abc.abstractmethod
    def evaluate_assignment(self, rated_votes: pd.DataFrame, assignments: pd.DataFrame) -> bool:
        """
        Evaluate if the assignment satisfies the axiom.
        """
        pass

    @abc.abstractmethod
    def satisfactory_slates(self, rated_votes: pd.DataFrame) -> set[frozenset[str]]:
        """
        Get the set of slates which satisfy the axiom.
        """
        pass
