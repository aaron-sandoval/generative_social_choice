from typing import Optional

import pandas as pd
import numpy as np
from jaxtyping import Float, Bool


def pareto_dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return all(a[i] >= b[i] for i in range(len(a))) and any(a[i] > b[i] for i in range(len(a)))


def is_pareto_efficient(utilities: Float[np.ndarray, "candidate utility_type"]) -> Bool[np.ndarray, "candidate"]:
    """
    Finds the boolean mask of pareto efficiency among an array of candidates.

    Higher utilities are better for all columns.
    If this is not the case for some metric, invert/negate that column before calling this function.
    """
    is_efficient = np.ones(utilities.shape[0], dtype = bool)
    for i, u in enumerate(utilities):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(utilities[is_efficient]>u, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

def voter_utilities(rated_votes: pd.DataFrame, assignments: pd.DataFrame, column_name: str = "utility") -> pd.Series:
    """
    Get the utility of each voter for a given assignment.
    """
    utilities = np.diag(rated_votes.loc[assignments.index, assignments["candidate_id"]])
    print(f"utilities: {utilities}")
    return pd.Series(utilities, index=assignments.index, name=column_name)


def voter_max_utilities_from_slate(rated_votes: pd.DataFrame, slate: set[str]) -> pd.Series:
    return rated_votes.loc[:, slate].max(axis=1)


def total_utility(rated_votes: pd.DataFrame, assignments: pd.DataFrame) -> float:
    """
    Get the total utility of a given assignment.
    """
    return voter_utilities(rated_votes, assignments).sum()


def mth_highest_utility(rated_votes: pd.DataFrame, assignments: pd.DataFrame, m: int) -> pd.Series:
    """
    Get the utility of the mth highest utility voter for a given assignment.

    # Returns
    - A length-1 Series with the voter ID and their utility.
    """
    return voter_utilities(rated_votes, assignments).nlargest(m).tail(1)


def min_utility(rated_votes: pd.DataFrame, assignments: pd.DataFrame) -> pd.Series:
    """
    Get the utility of the least happy voter for a given assignment.

    # Returns
    - A length-1 Series with the voter ID and their utility.
    """
    return voter_utilities(rated_votes, assignments).nsmallest(1)

