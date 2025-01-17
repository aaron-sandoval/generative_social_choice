from typing import Optional, Sequence, Callable
import itertools
import pandas as pd
import numpy as np
from jaxtyping import Float, Bool


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


def pareto_dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return all(a[i] >= b[i] for i in range(len(a))) and any(a[i] > b[i] for i in range(len(a)))


def is_pareto_efficient(utilities: Float[np.ndarray, "candidate utility_type"]) -> Bool[np.ndarray, "candidate"]:
    """
    Finds the boolean mask of pareto efficiency among an array of candidates.

    Higher utilities are better for all columns.
    If this is not the case for some metric, invert/negate that column before calling this function.
    Source: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    is_efficient = np.ones(utilities.shape[0], dtype = bool)
    for i, u in enumerate(utilities):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(utilities[is_efficient]>u, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


def pareto_efficient_slates(
    rated_votes: pd.DataFrame,
    slate_size: int, 
    positive_metrics: Sequence[Callable[[Float[np.ndarray, "voter_utility"]], float]]
) -> set[frozenset[str]]:
    """
    Find all pareto efficient slates of a given size according to a set of positive metrics.


    This function assumes that voters are assigned to their highest utility candidate in the slate.

    # Arguments
    - `rated_votes: pd.DataFrame`: The utility of each voter for each candidate
    - `slate_size: int`: The number of candidates to be selected
    - `positive_metrics: Sequence[Callable[[Float[np.ndarray, "voter_utility"]], float]]`: The metrics to be maximized
      - The metrics are a function only of a 1D array of voter utilities.
      - Support for metrics which are a function of additional arguments beyond this 1D array is not supported.
      - The metrics must all be defined such that higher valued are better.
      - If this is not the case for some metric, use an inversion/negation within the `Callable`.
    

    # Returns
    - A set of frozensets of candidate IDs that are pareto efficient.
    """
    metric_values: Float[np.ndarray, "slate metric_type"] = pd.DataFrame(
        index=itertools.combinations(rated_votes.columns, r=slate_size),
        columns=positive_metrics
    )
    for slate in metric_values.index:
        utilities = voter_max_utilities_from_slate(rated_votes, slate)
        for metric in positive_metrics:
            metric_values.loc[slate, metric] = metric(utilities)
    
    return set(metric_values.index[is_pareto_efficient(metric_values.values)])

