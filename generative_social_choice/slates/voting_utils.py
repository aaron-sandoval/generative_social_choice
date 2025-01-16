from typing import Optional

import pandas as pd
import numpy as np


def voter_utilities(rated_votes: pd.DataFrame, assignments: pd.DataFrame, column_name: str = "utility") -> pd.Series:
    """
    Get the utility of each voter for a given assignment.
    """
    utilities = np.diag(rated_votes.loc[assignments.index, assignments["candidate_id"]])
    return pd.Series(utilities, index=assignments.index, name=column_name)


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

