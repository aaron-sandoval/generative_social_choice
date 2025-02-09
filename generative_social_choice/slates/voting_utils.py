from collections.abc import Iterable
from typing import Optional, Sequence, Callable, Hashable
import itertools

import pandas as pd
import numpy as np
from jaxtyping import Float, Bool


def voter_utilities(rated_votes: pd.DataFrame, assignments_series: pd.Series, output_column_name: str = "utility") -> pd.Series:
    """
    Get the utility of each voter for a given assignment.
    """
    utilities = np.diag(rated_votes.loc[assignments_series.index, assignments_series.values])
    return pd.Series(utilities, index=assignments_series.index, name=output_column_name)


def voter_max_utilities_from_slate(rated_votes: pd.DataFrame, slate: set[str]) -> pd.Series:
    """
    Get the maximum possible utility of each voter within a given slate.
    """
    max_utilities = rated_votes.loc[:, slate].max(axis=1)
    max_candidates = rated_votes.loc[:, slate].idxmax(axis=1)
    return pd.DataFrame({
        "candidate_id": max_candidates,
        "utility": max_utilities
    }, index=rated_votes.index)


def total_utility(rated_votes: pd.DataFrame, assignments: pd.DataFrame) -> float:
    """
    Get the total utility of a given assignment.
    """
    return voter_utilities(rated_votes, assignments["candidate_id"]).sum()


def mth_highest_utility(rated_votes: pd.DataFrame, assignments: pd.DataFrame, m: int) -> pd.Series:
    """
    Get the utility of the mth highest utility voter for a given assignment.

    # Returns
    - A length-1 Series with the voter ID and their utility.
    """
    return voter_utilities(rated_votes, assignments["candidate_id"]).nlargest(m).tail(1)


def min_utility(rated_votes: pd.DataFrame, assignments: pd.DataFrame) -> pd.Series:
    """
    Get the utility of the least happy voter for a given assignment.

    # Returns
    - A length-1 Series with the voter ID and their utility.
    """
    return voter_utilities(rated_votes, assignments["candidate_id"]).nsmallest(1)


def pareto_dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    return all(a[i] >= b[i] for i in range(len(a))) and any(a[i] > b[i] for i in range(len(a)))


def is_pareto_efficient(positive_metrics: Float[np.ndarray, "slate metric_type"], abs_tol: float = 1e-6) -> Bool[np.ndarray, "slate"]:
    """
    Finds the boolean mask of pareto efficiency among an array of candidates.

    Higher utilities must be better for all metrics.
    If this is not the case for some metric, invert/negate that column before calling this function.
    Source: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python

    # Arguments
    - `positive_metrics: Float[np.ndarray, "slate metric_type"]`: The metrics to be maximized
    - `abs_tol: float`: Absolute tolerance for floating point comparisons
      - A metric a is considered to be greater than another metric b if a > b + abs_tol for all metrics.
      - If two slates have metrics which are all within abs_tol of each other, both are considered efficient.
    """
    is_efficient = np.ones(positive_metrics.shape[0], dtype=bool)
    for i, u in enumerate(positive_metrics):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(positive_metrics[is_efficient] > u + abs_tol, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self

    # Efficiently find pairs of slates where each pair of values in all columns is within abs_tol
    efficient_indices = np.where(is_efficient)[0]
    non_efficient_indices = np.where(~is_efficient)[0]

    for i in efficient_indices:
        for j in non_efficient_indices:
            if is_efficient[j]:  # Skip if a previous iteration already found j to be efficient
                continue
            if np.all(np.abs(positive_metrics[i] - positive_metrics[j]) <= abs_tol):
                is_efficient[j] = True

    return is_efficient


def pareto_efficient_slates(
    rated_votes: pd.DataFrame,
    slate_size: int, 
    positive_metrics: Iterable[Callable[[Float[np.ndarray, "voter_utility"]], float]]
) -> set[frozenset[Hashable]]:
    """
    Find all pareto efficient slates of a given size according to a set of positive metrics.


    This function assumes that voters are assigned to their highest utility candidate in the slate.

    # Arguments
    - `rated_votes: pd.DataFrame`: The utility of each voter for each candidate
    - `slate_size: int`: The number of candidates to be selected
    - `positive_metrics: Iterable[Callable[[Float[np.ndarray, "voter_utility"]], float]]`: The metrics to be maximized
      - The metrics are a function only of a 1D array of voter utilities.
      - Support for metrics which are a function of additional arguments beyond this 1D array is not supported.
      - The metrics must all be defined such that higher valued are better.
      - If this is not the case for some metric, use an inversion/negation within the `Callable`.
    

    # Returns
    - A set of frozensets of candidate IDs that are pareto efficient.
    """
    metric_values: Float[np.ndarray, "slate metric_type"] = pd.DataFrame(
        index=itertools.combinations(rated_votes.columns, r=slate_size),
        columns=range(len(positive_metrics)),
        dtype=float
    )
    # Could parallelize this if needed
    for slate in metric_values.index:
        utilities = voter_max_utilities_from_slate(rated_votes, slate)["utility"]
        for metric_index, metric in enumerate(positive_metrics):
            metric_values.at[slate, metric_index] = metric(utilities)
    
    return set(frozenset(cand_tuple) for cand_tuple in metric_values.index[is_pareto_efficient(metric_values.values)])


def gini(utilities: Float[np.ndarray, "person"], weights: Optional[Float[np.ndarray, "person"]] = None) -> float:
    """
    Calculate the Gini coefficient of a given array of utilities.


    Source: https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python

    # Arguments
    - `utilities: Float[np.ndarray, "person"]`: The utilities of the voters
    - `weights: Optional[Float[np.ndarray, "person"]]`: The weights of the voters
    

    """
    utilities = np.asarray(utilities)
    if weights is not None:

        weights = np.asarray(weights)
        sorted_indices = np.argsort(utilities)
        sorted_x = utilities[sorted_indices]
        sorted_w = weights[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) / 
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(utilities)
        n = len(utilities)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n