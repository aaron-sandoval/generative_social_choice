import abc
from functools import cache, wraps
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Hashable, Literal
import itertools

import pandas as pd
import numpy as np
from jaxtyping import Float, Bool
from kiwiutils.kiwilib import flatten

@dataclass(frozen=True)
class NoiseAugmentationMethod(abc.ABC):
    """
    A method for augmenting a single RatedVoteCase by adding extra cases with noise.
    """
    min_magnitude: Optional[float] = 1e-3
    max_magnitude: Optional[float] = 1e-1
    sign: Literal["positive", "negative", "both"] = "both"
    seed: Optional[int] = None

    
    @abc.abstractmethod
    def noise_dfs(self, rated_votes: pd.DataFrame, num_augmentations: int) -> list[pd.DataFrame]:
        """
        The noise dfs to add to the rated votes.
        """
        raise NotImplementedError
    
    def augment(self, rated_votes: pd.DataFrame, num_augmentations: int) -> list[pd.DataFrame]:
        """
        Augment the given rated votes with noise.

        # Arguments
        - `rated_votes: pd.DataFrame`: The rated votes to augment
        - `num_augmentations: int`: The number of additional cases to create

        # Returns
        - A list of `num_augmentations` pd.DataFrames, each with the same rated votes as the input but with noise added.
        """
        return [rated_votes + df for df in self.noise_dfs(rated_votes, num_augmentations)]

    def sample_signs(self, num_augmentations: int) -> list[int]:
        """
        Sample a list of signs for the noise.
        """
        if self.sign == "positive":
            return np.ones(num_augmentations, dtype=int)
        elif self.sign == "negative":
            return -np.ones(num_augmentations, dtype=int)
        else:  # "both"
            rng = np.random.default_rng(self.seed)
            return rng.choice([-1, 1], size=num_augmentations, p=[0.5, 0.5])


@dataclass(frozen=True)
class CellWiseAugmentation(NoiseAugmentationMethod):
    """
    Adds noise to each cell of the utility matrix independently.
    """
    def noise_dfs(self, rated_votes: pd.DataFrame, num_augmentations: int) -> list[pd.DataFrame]:
        """
        Adds noise to each cell of the utility matrix independently.
        """
        rng = np.random.default_rng(self.seed)
        signs = self.sample_signs(num_augmentations)
        augmented_votes = []
        
        for i in range(num_augmentations):
            noise = rng.uniform(self.min_magnitude, self.max_magnitude, rated_votes.shape)
            augmented_votes.append(signs[i] * noise)
            
        return augmented_votes

@dataclass(frozen=True)
class CandidateWiseAugmentation(NoiseAugmentationMethod):
    """
    Adds noise with a single value sampled for each candidate (column) and applied to all ratings for that candidate.
    """
    def noise_dfs(self, rated_votes: pd.DataFrame, num_augmentations: int) -> list[pd.DataFrame]:
        """
        Adds noise with a single value sampled for each candidate (column) and applied to all ratings for that candidate.
        
        # Arguments
        - `rated_votes: pd.DataFrame`: The rated votes to augment
        - `num_augmentations: int`: The number of additional cases to create
        
        # Returns
        - A list of `num_augmentations` pd.DataFrames, each with the same rated votes as the input but with noise added.
        """
        signs = self.sample_signs(num_augmentations)
        augmented_votes = []
        
        rng = np.random.default_rng(self.seed)
        
        for i in range(num_augmentations):
            # Sample one noise value per candidate (column)
            candidate_noise = rng.uniform(
                self.min_magnitude, 
                self.max_magnitude, 
                len(rated_votes.columns)
            )
            
            # Create a DataFrame with the same shape as rated_votes
            # Each column has the same noise value applied to all rows
            noise_df = pd.DataFrame(
                {col: candidate_noise[j] for j, col in enumerate(rated_votes.columns)},
                index=rated_votes.index
            )
            
            # Apply the noise with the appropriate sign
            augmented_votes.append(signs[i] * noise_df)
            
        return augmented_votes


class VoterWiseAugmentation(NoiseAugmentationMethod):
    """
    Adds noise with a single value sampled for each voter (row) and applied to all ratings for that voter.
    """
    def noise_dfs(self, rated_votes: pd.DataFrame, num_augmentations: int) -> list[pd.DataFrame]:
        """
        Adds noise with a single value sampled for each voter (row) and applied to all ratings for that voter.
        
        # Arguments
        - `rated_votes: pd.DataFrame`: The rated votes to augment
        - `num_augmentations: int`: The number of additional cases to create
        
        # Returns
        - A list of `num_augmentations` pd.DataFrames, each with the same rated votes as the input but with noise added.
        """
        signs = self.sample_signs(num_augmentations)
        augmented_votes = []
        
        rng = np.random.default_rng(self.seed)
        
        for i in range(num_augmentations):
            # Sample one noise value per voter (row)
            voter_noise = rng.uniform(
                self.min_magnitude, 
                self.max_magnitude, 
                len(rated_votes.index)
            )
            
            # Create a DataFrame with the same shape as rated_votes
            # Each row has the same noise value applied to all columns
            noise_df = pd.DataFrame(
                {col: voter_noise for col in rated_votes.columns},
                index=rated_votes.index
            )
            
            # Apply the noise with the appropriate sign
            augmented_votes.append(signs[i] * noise_df)
            
        return augmented_votes

class VoterAndCellWiseAugmentation(NoiseAugmentationMethod):
    """
    Adds noise with a single value sampled for each voter (row) and each cell of the utility matrix.
    """
    def noise_dfs(self, rated_votes: pd.DataFrame, num_augmentations: int) -> list[pd.DataFrame]:
        """
        Adds noise with a single value sampled for each voter (row) and each cell of the utility matrix.
        """
        # Create sub-augmentations with derived seeds to maintain independence
        voter_seed = None if self.seed is None else self.seed * 2
        cell_seed = None if self.seed is None else self.seed * 2 + 1
        
        voter_wise_augmentation = VoterWiseAugmentation(
            min_magnitude=self.max_magnitude*.5, 
            max_magnitude=self.max_magnitude - self.min_magnitude,
            seed=voter_seed
        )
        cell_wise_augmentation = CellWiseAugmentation(
            min_magnitude=self.min_magnitude, 
            max_magnitude=self.max_magnitude*.5,
            seed=cell_seed
        )
        return [a + b for a, b in zip(voter_wise_augmentation.noise_dfs(rated_votes, num_augmentations), cell_wise_augmentation.noise_dfs(rated_votes, num_augmentations))]
        

def voter_utilities(rated_votes: pd.DataFrame, assignments_series: pd.Series, output_column_name: str = "utility") -> pd.Series:
    """
    Get the utility of each voter for a given assignment.
    """
    utilities = np.diag(rated_votes.loc[assignments_series.index, assignments_series.values])
    return pd.Series(utilities, index=assignments_series.index, name=output_column_name)


def voter_max_utilities_from_slate(rated_votes: pd.DataFrame, slate: list[str]) -> pd.Series:
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


def is_pareto_efficient(positive_metrics: Float[np.ndarray, "slate metric_type"], abs_tol: float = 1e-6) -> Bool[np.ndarray, "slate"]:  # noqa: F821
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
    positive_metrics: Iterable[Callable[[Float[np.ndarray, "voter_utility"]], float]]  # noqa: F821
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


def generalized_lorenz_curve(utilities: Float[np.ndarray, "voter"]) -> Float[np.ndarray, "voter"]:  # noqa: F821
    """
    Compute the generalized Lorenz curve for a given utility vector.


    The generalized Lorenz curve is the cumulative sum of utilities sorted in
    ascending order.

    # Arguments
    - `utilities: Float[np.ndarray, "voter"]`: The utility vector

    # Returns
    - `Float[np.ndarray, "voter"]`: The cumulative sums of sorted utilities
    """
    sorted_utilities = np.sort(utilities)
    return np.cumsum(sorted_utilities)


def gini(utilities: Float[np.ndarray, "person"], weights: Optional[Float[np.ndarray, "person"]] = None) -> float:  # noqa: F821
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
    


## Option 3: Use a custom key function with lru_cache


def df_cache(func):
    """Cache decorator that handles DataFrames by using their content hash."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create hashable versions of args and kwargs
        hashable_args = []
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                # Use pandas' built-in hash function for the dataframe content
                hashable_args.append(hash(pd.util.hash_pandas_object(arg).sum()))
            else:
                hashable_args.append(arg)
        
        hashable_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, pd.DataFrame):
                hashable_kwargs[key] = hash(pd.util.hash_pandas_object(value).sum())
            else:
                hashable_kwargs[key] = value
        
        # Use the cached function with hashable arguments
        return _cached_func(tuple(hashable_args), frozenset(hashable_kwargs.items()))
    
    # The actual cached function that takes hashable arguments
    @cache
    def _cached_func(hashable_args, hashable_kwargs):
        # Convert back to the original args and kwargs
        args_list = list(hashable_args)
        kwargs_dict = dict(hashable_kwargs)
        
        # Call the original function
        return func(*args_list, **kwargs_dict)
    
    return wrapper


def filter_candidates_by_individual_pareto_efficiency(rated_votes: pd.DataFrame) -> pd.DataFrame:
    """
    Filter candidates by individual Pareto efficiency.
    
    This function removes candidates that are individually Pareto dominated by other 
    candidates. A candidate is individually Pareto efficient if there is no other 
    candidate that provides equal or better utility for all voters and strictly 
    better utility for at least one voter.
    
    # Arguments
    - `rated_votes: pd.DataFrame`: The utility of each voter (rows) for each candidate (columns)
    
    # Returns
    - `pd.DataFrame`: A DataFrame with the same structure as the input but with only 
      individually Pareto efficient candidates (columns) retained.
    """
    if len(rated_votes.columns) == 0:
        return rated_votes
    efficient_candidates = set(flatten(pareto_efficient_slates(rated_votes, 1, [(lambda utilities, idx=i: utilities[idx]) for i in range(len(rated_votes))])))
    # Preserve the original column order
    ordered_efficient_candidates = [col for col in rated_votes.columns if col in efficient_candidates]
    return rated_votes.loc[:, ordered_efficient_candidates]
