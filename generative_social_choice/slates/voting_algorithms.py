from pathlib import Path
from dataclasses import dataclass
import abc
from typing import Optional, Literal, override

import pandas as pd
import numpy as np

from generative_social_choice.utils.helper_functions import (
    get_time_string,
    get_base_dir_path,
)
from generative_social_choice.queries.query_interface import Agent, Generator


Phragmen_Load_Magnitude = Literal["marginal_slate", "marginal_previous", "total"]
"""
The magnitude of the load added for a new candidate.
- "marginal_slate": The load added is 1/the marginal utility created by the new candidate compared to the best alternative in the slate for each voter.
  - When reassigning, the total load of the candidate is set by the sum of marginal utilities created by that candidate compared to the best alternative in the slate for each voter.
- "marginal_previous": The load added is 1/the marginal utility created by the new candidate only among reassigned voters.
  - When reassigning, the total load of the candidate is set by the sum of marginal utilities created by that candidate compared to the candidate to which each voter was previously assigned.
- "total": The load added is 1/the total utility created by the new candidate among reassigned voters.
"""


BASELINE_UTILITY = 0.0  # Assumed utility of a voter unassigned/unrepresented
NULL_CANDIDATE_ID = "NULL_CAND"


@dataclass(frozen=True)
class VotingAlgorithm(abc.ABC):
    """
    Abstract base class for a voting algorithm.
    """
    @abc.abstractmethod
    def vote(
        self,
        rated_votes: pd.DataFrame,
        slate_size: int
    ) -> tuple[list[str], pd.DataFrame]:
        """
        Select a slate of candidates and assign them to the voters.

        # Arguments
        - `rated_votes: pd.DataFrame`: Utility of each voter (rows) for each candidate (columns)
        - `slate_size: int`: The number of candidates to be selected

        # Returns
        - `slate: list[str]`: The slate of candidates to be selected
        - `assignments: pd.DataFrame`: The assignments of the candidates to the voters with the following columns:
            - `candidate_id`: The candidate to which the voter is assigned
            - Various other columns may be present depending on the algorithm
        """
        pass

    @property
    def name(self) -> str:
        """
        Succinct name of the voting algorithm, used in labeling test cases.
        """
        return repr(self)


@dataclass(frozen=True)
class TunableVotingAlgorithm(VotingAlgorithm):
    """
    A voting algorithm whose egalitarian-utilitarian trade-off can be tuned by a hyperparameter.

    # Arguments
    - `egalitarian_utilitarian: float = 1.0`: Hyperparameter governing the egalitarian-utilitarian trade-off.
      - 0.0: Egalitarian objective: Maximize the minimum utility among all individual voters
      - 1.0: Utilitarian objective: Maximize the total utility among all individual voters
    """
    egalitarian_utilitarian: float = 0.5


@dataclass(frozen=True)
class SequentialPhragmenMinimax(VotingAlgorithm):
    load_magnitude_method: Phragmen_Load_Magnitude = "marginal_slate"
    clear_reassigned_loads: bool = True
    redistribute_defected_candidate_loads: bool = True

    @property
    def name(self) -> str:
        return f"Phragmen({self.load_magnitude_method}, clear={self.clear_reassigned_loads}, redistr={self.redistribute_defected_candidate_loads})"

    @override
    def vote(
        self,
        rated_votes: pd.DataFrame,
        slate_size: int,
    ) -> tuple[list[str], pd.DataFrame]:
        """
        # Returns
        - `slate: List[str]`: The slate of candidates to be selected
        - `assignments: pd.DataFrame`: The assignments of the candidates to the voters with the following columns:
            - `candidate_id`: GUARANTEED: The candidate to which the voter is assigned
            - Other columns are returned for debugging and unit testing purposes, not guaranteed to always be present
            - `load`: The load of the candidate
            - `utility`: The utility of the voter for the candidate
            - `utility_previous`: The utility of the voter for the candidate before the current assignment
            - `second_selected_candidate_id`: The 2nd-favorite candidate for each voter among the candidates in the current slate
        """
        # TODO: Figure out egalitarian_utilitarian

        # Initialize the slate and assignments
        slate: list[str] = []
        rejected_candidates: set[str] = {NULL_CANDIDATE_ID}  # Candidates that have been added to the slate and later had all their voters defect
        assignments: pd.DataFrame = pd.DataFrame(index=rated_votes.index)
        cols={"candidate_id": str, "load": float, "utility": float, "utility_previous": float, "second_selected_candidate_id": str}
        for col, dtype in cols.items():
            assignments[col] = pd.Series(index=assignments.index, dtype=dtype)
        # "second_selected_candidate_id": the 2nd-favorite candidate for each voter among the candidates in the current slate
        
        assignments["load"] = 0.0
        assignments["utility_previous"] = BASELINE_UTILITY
        assignments["utility"] = BASELINE_UTILITY
        assignments["second_selected_candidate_id"] = NULL_CANDIDATE_ID
        assignments["candidate_id"] = NULL_CANDIDATE_ID
        rated_votes[NULL_CANDIDATE_ID] = BASELINE_UTILITY  # Append a column for the null candidate

        i = 0
        valid_candidates = set(rated_votes.columns) - rejected_candidates

        while len(slate) < slate_size and i <= len(rated_votes.columns)+1:
            i += 1
            min_load = float("inf")
            min_load_candidate_id: int = NULL_CANDIDATE_ID
            # min_load_assignments = assignments.copy()
            for candidate in valid_candidates:
                if candidate == NULL_CANDIDATE_ID:
                    continue
                assignments_with_candidate = self._phragmen_update_assignments(
                    assignments,
                    rated_votes,
                    candidate,
                )
                if assignments_with_candidate["load"].min() < min_load:
                    min_load = assignments_with_candidate["load"].min()
                    min_load_candidate_id = candidate
                    min_load_assignments = assignments_with_candidate.copy()

            slate.append(min_load_candidate_id)
            assignments = min_load_assignments
            rejected_candidates.update(set(slate) ^ set(assignments["candidate_id"]))
            valid_candidates -= rejected_candidates.union(slate)  # Remove candidates not to be considered in the next iteration
        
        # Remove the null candidate column in case the modification would persist outside the function
        rated_votes = rated_votes.drop(columns=[NULL_CANDIDATE_ID])

        return slate, assignments

    def _phragmen_update_assignments(
        self,
        assignments: pd.DataFrame,
        rated_votes: pd.DataFrame,
        candidate: int,
    ) -> pd.DataFrame:
        """
        Update the assignments of the voters to the candidates.

        # Arguments
        - `assignments: pd.DataFrame`: The assignments of the candidates to the voters
        - `rated_votes: pd.DataFrame`: Utility of each voter (rows) for each candidate (columns)
        - `candidate: int`: The candidate to be added to the slate
        - `load_magnitude_method: Phragmen_Load_Magnitude`: The method to use to determine the load magnitude
        - `clear_reassigned_loads: bool`: Whether to clear the loads of the reassigned voters or add them to the loads accrued from previous assignments.
        """
        new_assignments = assignments.copy()
        is_reassigned = rated_votes[candidate] > new_assignments["utility"] # It matters if this is > or >= for int-valued utilities
        reassignments = assignments.loc[is_reassigned]
        old_assignments = assignments.loc[is_reassigned]
        new_assignments.loc[is_reassigned, "second_selected_candidate_id"] = reassignments["candidate_id"]
        new_assignments.loc[is_reassigned, "candidate_id"] = candidate
        new_assignments.loc[is_reassigned, "utility_previous"] = reassignments["utility"] # TODO: Check that this makes a deep copy
        new_assignments.loc[is_reassigned, "utility"] = rated_votes[candidate]

        if any(is_reassigned):
            if self.load_magnitude_method == "marginal_previous":
                new_candidate_total_load: float = 1 / (rated_votes[is_reassigned, candidate] - old_assignments["utility"]).sum()
            elif self.load_magnitude_method == "marginal_slate":
                # is_first_assignment_voters = new_assignments.loc[is_reassigned, "second_selected_candidate_id"].isna()
                marginal_utility = new_assignments.loc[is_reassigned, "utility"]
                next_best_utility = np.diag(rated_votes.loc[reassignments.index, new_assignments.loc[is_reassigned, "second_selected_candidate_id"]].to_numpy())
                marginal_utility -= pd.Series(next_best_utility, index=reassignments.index).fillna(BASELINE_UTILITY)
                new_candidate_total_load: float = 1 / (marginal_utility.sum())

                # Update 2nd-favorite candidate for each voter
                # https://pandas.pydata.org/docs/user_guide/indexing.html#looking-up-values-by-index-column-labels
                rated_votes["second_fav_cand_id"] = new_assignments.second_selected_candidate_id
                idx, cols = pd.factorize(rated_votes["second_fav_cand_id"])
                cur_2nd_fav_utility = pd.Series(rated_votes.reindex(cols, axis=1).to_numpy()[np.arange(len(rated_votes)), idx], index=rated_votes.index)
                new_2nd_favorite_voters = (rated_votes[candidate] >= cur_2nd_fav_utility) & ~is_reassigned
                new_assignments.loc[new_2nd_favorite_voters, "second_selected_candidate_id"] = candidate
            elif self.load_magnitude_method == "total":
                new_candidate_total_load: float = 1 / (rated_votes.loc[is_reassigned, candidate].sum())
            else:
                raise ValueError(f"Invalid load magnitude method: {self.load_magnitude_method}")
        
        # Calculate the loads for the reassigned voters
        new_candidate_loads: pd.Series = rated_votes.loc[is_reassigned, candidate] * new_candidate_total_load**2
        if self.clear_reassigned_loads:
            new_assignments.loc[is_reassigned, "load"] = new_candidate_loads
        else:
            new_assignments.loc[is_reassigned, "load"] += new_candidate_loads
        
        if self.redistribute_defected_candidate_loads:
            affected_candidates = old_assignments.loc[is_reassigned, "candidate_id"].unique()
            for affected_cand in affected_candidates:
                if affected_cand == NULL_CANDIDATE_ID:
                    continue
                affected_cand_voters = new_assignments[new_assignments["candidate_id"] == affected_cand]
                if len(affected_cand_voters) == 0:
                    continue
                if self.load_magnitude_method == "marginal_previous":
                    marginal_utility = affected_cand_voters["utility"] - affected_cand_voters["utility_previous"]
                    # defected_cand_total_load = 1 / (affected_cand_voters["utility_previous"] - old_assignments.loc[old_assignments["candidate_id"] == affected_cand, "utility_previous"]).sum()
                elif self.load_magnitude_method == "marginal_slate":
                    # is_first_assignment_voters = affected_cand_voters.loc[affected_cand_voters["second_selected_candidate_id"]].isna()
                    marginal_utility = affected_cand_voters["utility"] 
                    marginal_utility -= affected_cand_voters["second_selected_candidate_id"]
                    # marginal_utility[is_first_assignment_voters] -= BASELINE_UTILITY
                elif self.load_magnitude_method == "total":
                    marginal_utility = affected_cand_voters["utility"]

                # Redistribute the loads
                defected_cand_total_load = 1 / (marginal_utility.sum())
                new_assignments.loc[affected_cand_voters.index, "load"] = marginal_utility / defected_cand_total_load**2

        return new_assignments
