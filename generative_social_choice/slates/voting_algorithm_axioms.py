import abc
import itertools
from dataclasses import dataclass, field
from typing import override

import pandas as pd
import numpy as np
from jaxtyping import Float

from generative_social_choice.slates.voting_utils import (
    voter_utilities,
    voter_max_utilities_from_slate,
    pareto_dominates,
    pareto_efficient_slates
)


@dataclass(frozen=True)
class VotingAlgorithmAxiom(abc.ABC):
    """
    An axiom which a voting algorithm may satisfy.

    An algorithm satisfies an axiom if it satisfies the axiom for all possible rated votes and slate sizes.
    """
    name: str

    def __post_init__(self):
        if not self.name:
            object.__setattr__(self, "name", type(self).__name__)

    @abc.abstractmethod
    def evaluate_assignment(self, rated_votes: pd.DataFrame, slate_size: int, assignments: pd.DataFrame) -> bool:
        """
        Evaluate if the assignment satisfies the axiom.
        """
        pass

    @abc.abstractmethod
    def satisfactory_slates(self, rated_votes: pd.DataFrame, slate_size: int) -> set[frozenset[str]]:
        """
        Get the set of slates which satisfy the axiom.
        """
        pass


@dataclass(frozen=True)
class NonRadicalAxiom(VotingAlgorithmAxiom):
    """
    ABC for axioms governing allowable tradeoffs between two metrics.

    # Arguments
    - `max_tradeoff`: A maximum allowable tradeoff ratio between two metrics.
    """
    max_tradeoff: float


@dataclass(frozen=True)
class IndividualParetoAxiom(VotingAlgorithmAxiom):
    """For all solutions, there is no slate for which total utility strictly improves and for no member the utility decreases."""

    name: str = "Individual Pareto Efficiency"

    @override
    def evaluate_assignment(self, rated_votes: pd.DataFrame, slate_size: int, assignments: pd.DataFrame) -> bool:
        # Get utilities for the given assignments
        w_utilities = np.array(voter_utilities(rated_votes, assignments["candidate_id"]))

        for Wprime in itertools.combinations(rated_votes.columns, r=slate_size):
            # Compute utilities (using optimal assignment for given slate)
            wprime_utilities = rated_votes.loc[:, Wprime].max(axis=1).to_numpy()

            # There is no slate for which total utility strictly improves and for no member the utility decreases
            if wprime_utilities.sum() > w_utilities.sum() and (wprime_utilities >= w_utilities).all():
                return False
        return True
    
    @override
    def satisfactory_slates(self, rated_votes: pd.DataFrame, slate_size: int) -> set[frozenset[str]]:
        efficient_slates = []  # (slate, total utility, utilities)

        for slate in itertools.combinations(rated_votes.columns, r=slate_size):
            # Compute utilities (using optimal assignment for given slate)
            utilities = rated_votes.loc[:, slate].max(axis=1).to_numpy()
            total_utility = utilities.sum()

            no_better_slate_exists = True
            for other_slate, other_total_utility, other_utilities in efficient_slates[:]:
                # If the new slate is strictly better, we drop the old one
                if other_total_utility < total_utility and (other_utilities <= utilities).all():
                    efficient_slates.remove( (other_slate, other_total_utility, other_utilities) )

                # If strictly better combinations exist, this slate is not interesting
                if other_total_utility > total_utility and (other_utilities >= utilities).all():
                    no_better_slate_exists = False
                    break

            if no_better_slate_exists:
                efficient_slates.append( (slate, total_utility, utilities) )

        return [slate[0] for slate in efficient_slates]


@dataclass(frozen=True)            
class HappiestParetoAxiom(VotingAlgorithmAxiom):
    """No other slate has m-th happiest person at least as good for all m and strictly better for at least one m*.
    
    Note that we get the m-th happiest person vector by sorting the utilities in descending order."""

    name: str = "m-th Happiest Person Pareto Efficiency"

    @override
    def evaluate_assignment(self, rated_votes: pd.DataFrame, slate_size: int, assignments: pd.DataFrame) -> bool:
        # Get utilities for the given assignments
        w_utilities = np.array(voter_utilities(rated_votes, assignments["candidate_id"]))

        for Wprime in itertools.combinations(rated_votes.columns, r=slate_size):
            # Compute utilities (using optimal assignment for given slate)
            wprime_utilities = rated_votes.loc[:, Wprime].max(axis=1).to_numpy()

            # No other slate has m-th happiest person at least as good for all m and strictly better for at least one m'
            # (Note that we get the m-th happiest person function by sorting the utilities in descending order.)
            mth_happiest = np.sort(w_utilities)[::-1]
            mth_happiest_prime = np.sort(wprime_utilities)[::-1]
            if (mth_happiest_prime > mth_happiest).any() and (mth_happiest_prime <= mth_happiest).all():
                return False
        return True
    
    @override
    def satisfactory_slates(self, rated_votes: pd.DataFrame, slate_size: int) -> set[frozenset[str]]:
        efficient_slates = []  # (slate, mth happiest person vector)

        for slate in itertools.combinations(rated_votes.columns, r=slate_size):
            # Compute utilities (using optimal assignment for given slate)
            utilities = rated_votes.loc[:, slate].max(axis=1).to_numpy()
            mth_happiest = np.sort(utilities)[::-1]

            no_better_slate_exists = True
            for other_slate, other_mth_happiest in efficient_slates[:]:
                # If the new slate is strictly better, we drop the old one
                if (other_mth_happiest < mth_happiest).any() and (other_mth_happiest >= mth_happiest).all():
                    efficient_slates.remove( (other_slate, other_mth_happiest) )

                # If strictly better combinations exist, this slate is not interesting
                if (other_mth_happiest > mth_happiest).any() and (other_mth_happiest <= mth_happiest).all():
                    no_better_slate_exists = False
                    break

            if no_better_slate_exists:
                efficient_slates.append( (slate, mth_happiest) )

        return [slate[0] for slate in efficient_slates]
    

@dataclass(frozen=True)
class CoverageAxiom(VotingAlgorithmAxiom):
    """Representing as many people as possible:
    There is no other slate with assignment wprime with at least the same total utility and a threshold m, such that
    - h(w,mprime)>=h(wprime,mprime) for all mprime>=m [h(w,mprime) is mprime-th happiest person under assignment w]
    - h(w,m*)>h(wprime,m*) for some m*>=m
    """

    name: str = "Maximum Coverage"

    @override
    def evaluate_assignment(self, rated_votes: pd.DataFrame, slate_size: int, assignments: pd.DataFrame) -> bool:
        # Get utilities for the given assignments
        w_utilities = np.array(voter_utilities(rated_votes, assignments["candidate_id"]))

        for Wprime in itertools.combinations(rated_votes.columns, r=slate_size):
            # Compute utilities (using optimal assignment for given slate)
            wprime_utilities = rated_votes.loc[:, Wprime].max(axis=1).to_numpy()

            # There is no other slate with at least the same total utility and a threshold m,
            # such that m'-th happiest person for that slate is >= for all m'>=m and > for some m*
            matching_total_utility = wprime_utilities.sum() >= w_utilities.sum()
            strictly_greater_ms = np.where(wprime_utilities > w_utilities)[0]
            if len(strictly_greater_ms) > 0:
                # If from this index on, m-th happiest person never has lower utility in w_prime, then the threshold is valid
                threshold_exists = (wprime_utilities[strictly_greater_ms.max():] < w_utilities[strictly_greater_ms.max():]).sum() == 0
                if matching_total_utility and threshold_exists:
                    return False
        return True
    
    @override
    def satisfactory_slates(self, rated_votes: pd.DataFrame, slate_size: int) -> set[frozenset[str]]:
        efficient_slates = []  # (slate, total_utility, utilities)

        for slate in itertools.combinations(rated_votes.columns, r=slate_size):
            # Compute utilities (using optimal assignment for given slate)
            utilities = rated_votes.loc[:, slate].max(axis=1).to_numpy()
            total_utility = utilities.sum()

            no_better_slate_exists = True
            for other_slate, other_total_utility, other_utilities in efficient_slates[:]:
                # If the new slate is strictly better, we drop the old one
                if other_total_utility <= total_utility:
                    strictly_greater_ms = np.where(other_utilities < utilities)[0]
                    if len(strictly_greater_ms) > 0:
                        threshold_exists = (other_utilities[strictly_greater_ms.max():] > utilities[strictly_greater_ms.max():]).sum() == 0
                        if threshold_exists:
                            efficient_slates.remove( (other_slate, other_total_utility, other_utilities) )

                # If strictly better combinations exist, this slate is not interesting
                if other_total_utility >= total_utility:
                    strictly_greater_ms = np.where(other_utilities > utilities)[0]
                    if len(strictly_greater_ms) > 0:
                        threshold_exists = (other_utilities[strictly_greater_ms.max():] < utilities[strictly_greater_ms.max():]).sum() == 0
                        if threshold_exists:
                            no_better_slate_exists = False
                            break

            if no_better_slate_exists:
                efficient_slates.append( (slate, total_utility, utilities) )

        return [slate[0] for slate in efficient_slates]
    

@dataclass(frozen=True)
class MinimumAndTotalUtilityParetoAxiom(VotingAlgorithmAxiom):
    """There is no other slate with strictly better minimum utility and total utility among individual voters.
    """

    name: str = "Minimum Utility and Total Utility Pareto Efficiency"

    @override
    def evaluate_assignment(self, rated_votes: pd.DataFrame, slate_size: int, assignments: pd.DataFrame) -> bool:
        # Get utilities for the given assignments
        w_utilities = np.array(voter_utilities(rated_votes, assignments["candidate_id"]))

        for Wprime in itertools.combinations(rated_votes.columns, r=slate_size):
            # Compute utilities (using optimal assignment for given slate)
            wprime_utilities = rated_votes.loc[:, Wprime].max(axis=1).to_numpy()
            if wprime_utilities.min() >= w_utilities.min() and wprime_utilities.sum() >= w_utilities.sum():
                if wprime_utilities.min() > w_utilities.min() or wprime_utilities.sum() > w_utilities.sum():
                    return False
        return True
    
    @override
    def satisfactory_slates(self, rated_votes: pd.DataFrame, slate_size: int) -> set[frozenset[str]]:
        return pareto_efficient_slates(rated_votes, slate_size, [lambda utilities: utilities.min(), lambda utilities: utilities.sum()])
        

@dataclass(frozen=True)
class NonRadicalTotalUtilityAxiom(NonRadicalAxiom):
    """
    There is no other slate with much higher min utility and slightly lower average utility.

    The assignment output of a voting algorithm has the minimum utility is u_min and the average utility u_avg. 
    The assignment meets the axiom if there exists no other slate with average utility = u_avg - delta and min utility u_min + epsilon, 
    where epsilon/delta >= `max_tradeoff` and epsilon > 0 and delta > 0.
    """
    
    max_tradeoff: float = 20.0
    name: str = "Non-radical Total Utility Pareto Efficiency"
    abs_tol: float = 1e-8

    @override
    def evaluate_assignment(self, rated_votes: pd.DataFrame, slate_size: int, assignments: pd.DataFrame) -> bool:
        def utility_tradeoff(alternate_utilities: Float[np.ndarray, "voter"]) -> float:
            if alternate_utilities.min() - self.abs_tol <= utilities.min() or alternate_utilities.mean() + self.abs_tol >= utilities.mean():
                return -1.0
            return (alternate_utilities.min() - utilities.min()) / (utilities.mean() - alternate_utilities.mean())

        utilities = voter_utilities(rated_votes, assignments["candidate_id"]).values
        worst_alt_slates: set[frozenset[str]] = pareto_efficient_slates(rated_votes, slate_size, [utility_tradeoff])

        for alt_slate in worst_alt_slates:
            alt_utilities = voter_max_utilities_from_slate(rated_votes, alt_slate)["utility"].values
            this_tradeoff = utility_tradeoff(alt_utilities)
            if this_tradeoff > self.max_tradeoff:
                return False
        return True

    @override
    def satisfactory_slates(self, rated_votes: pd.DataFrame, slate_size: int) -> set[frozenset[str]]:
        """
        Identify slates that satisfy the non-radical total utility axiom.
        """
        # Compute utility tuples for each slate
        slate_utilities = []
        for slate in itertools.combinations(rated_votes.columns, r=slate_size):
            utilities = rated_votes.loc[:, slate].max(axis=1).to_numpy()
            avg_utility = utilities.mean()
            min_utility = utilities.min()
            slate_utilities.append((frozenset(slate), avg_utility, min_utility))

        # Sort slates by average utility in descending order
        slate_utilities.sort(key=lambda x: x[1], reverse=True)

        # Start with all slates assumed valid
        valid_slates = {slate for slate, _, _ in slate_utilities}

        # Check each ordered pair (primary, alternate)
        for i, (primary_slate, primary_avg, primary_min) in enumerate(slate_utilities):
            if primary_slate not in valid_slates:
                continue

            for alternate_slate, alternate_avg, alternate_min in slate_utilities[i+1:]:
                epsilon = alternate_min - primary_min
                delta = primary_avg - alternate_avg

                # Check if epsilon and delta are valid and if the tradeoff exceeds max_tradeoff
                if epsilon > 0 and delta > 0 and (epsilon / delta) >= self.max_tradeoff:
                    valid_slates.discard(primary_slate)
                    break

        return valid_slates


@dataclass(frozen=True)
class NonRadicalMinUtilityAxiom(NonRadicalAxiom):
    """
    There is no other slate with much higher total utility and slightly lower minimum utility.

    The assignment output of a voting algorithm has the minimum utility u_min and the total utility u_total.
    The assignment meets the axiom if there exists no other slate with total utility = u_total + delta and min utility u_min - epsilon,
    where delta/epsilon >= `max_tradeoff` and epsilon > 0 and delta > 0.
    """
    
    max_tradeoff: float = 20.0
    name: str = "Non-radical Minimum Utility Pareto Efficiency"
    abs_tol: float = 1e-8

    def evaluate_assignment(self, rated_votes: pd.DataFrame, slate_size: int, assignments: pd.DataFrame) -> bool:
        def utility_tradeoff(alternate_utilities: Float[np.ndarray, "voter"]) -> float:
            if alternate_utilities.mean() - self.abs_tol <= utilities.mean() or alternate_utilities.min() + self.abs_tol >= utilities.min():
                return -1.0
            return (utilities.mean() - alternate_utilities.mean()) / (utilities.min() - alternate_utilities.min())

        utilities = voter_utilities(rated_votes, assignments["candidate_id"]).values
        worst_alt_slates: set[frozenset[str]] = pareto_efficient_slates(rated_votes, slate_size, [utility_tradeoff])

        for alt_slate in worst_alt_slates:
            alt_utilities = voter_max_utilities_from_slate(rated_votes, alt_slate)["utility"].values
            this_tradeoff = utility_tradeoff(alt_utilities)
            if this_tradeoff > self.max_tradeoff:
                return False
        return True

    def satisfactory_slates(self, rated_votes: pd.DataFrame, slate_size: int) -> set[frozenset[str]]:
        """
        Identify slates that satisfy the non-radical minimum utility axiom.
        """
        # Compute utility tuples for each slate
        slate_utilities = []
        for slate in itertools.combinations(rated_votes.columns, r=slate_size):
            utilities = rated_votes.loc[:, slate].max(axis=1).to_numpy()
            avg_utility = utilities.mean()
            min_utility = utilities.min()
            slate_utilities.append((frozenset(slate), avg_utility, min_utility))

        # Sort slates by minimum utility in ascending order
        slate_utilities.sort(key=lambda x: x[2])

        # Start with all slates assumed valid
        valid_slates = {slate for slate, _, _ in slate_utilities}

        # Check each ordered pair (primary, alternate)
        for i, (primary_slate, primary_avg, primary_min) in enumerate(slate_utilities):
            if primary_slate not in valid_slates:
                continue

            for alternate_slate, alternate_avg, alternate_min in slate_utilities[i+1:]:
                epsilon = primary_min - alternate_min
                delta = alternate_avg - primary_avg

                # Check if epsilon and delta are valid and if the tradeoff exceeds max_tradeoff
                if epsilon > 0 and delta > 0 and (delta / epsilon) >= self.max_tradeoff:
                    valid_slates.discard(primary_slate)
                    break

        return valid_slates



