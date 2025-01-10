from pathlib import Path
import pandas as pd
import os
from typing import Optional, Literal
from tqdm import tqdm
import concurrent.futures

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


def _phragmen_update_assignments(
    assignments: pd.DataFrame,
    rated_votes: pd.DataFrame,
    candidate: int,
    load_magnitude_method: Phragmen_Load_Magnitude = "marginal_slate",
    clear_reassigned_loads: bool = True,
    redistribute_defected_candidate_loads: bool = True,
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
    reassignments = assignments.loc[is_reassigned, :]
    old_assignments = assignments.loc[is_reassigned]
    reassignments["2nd_selected_candidate_id"] = reassignments["candidate_id"]
    reassignments["candidate_id"] = candidate
    reassignments["utility_previous"] = reassignments["utility"] # TODO: Check that this makes a deep copy
    reassignments["utility"] = rated_votes[candidate]

    if load_magnitude_method == "marginal_previous":
        new_candidate_total_load: float = 1 / (rated_votes[is_reassigned, candidate] - old_assignments["utility"]).sum()
    elif load_magnitude_method == "marginal_slate":
        is_first_assignment_voters = new_assignments.loc[new_assignments["2nd_selected_candidate_id"]].isna()
        marginal_utility = reassignments["utility"]
        marginal_utility -= new_assignments[~is_first_assignment_voters, "2nd_selected_candidate_id"]
        marginal_utility[is_first_assignment_voters] -= BASELINE_UTILITY
        new_candidate_total_load: float = 1 / (marginal_utility.sum())

        # Update 2nd-favorite candidate for each voter
        new_2nd_favorite_voters = rated_votes[~is_reassigned, candidate] >= new_assignments[~is_reassigned, "2nd_selected_candidate_id"]
        new_assignments.loc[new_2nd_favorite_voters, "2nd_selected_candidate_id"] = candidate
    elif load_magnitude_method == "total":
        new_candidate_total_load: float = 1 / (rated_votes[is_reassigned, candidate].sum())
    else:
        raise ValueError(f"Invalid load magnitude method: {load_magnitude_method}")
    
    # Calculate the loads for the reassigned voters
    new_candidate_loads: pd.Series = rated_votes[is_reassigned, candidate] / new_candidate_total_load**2
    if clear_reassigned_loads:
        new_assignments.loc[is_reassigned, "load"] = new_candidate_loads
    else:
        new_assignments.loc[is_reassigned, "load"] += new_candidate_loads
    
    if redistribute_defected_candidate_loads:
        affected_candidates = old_assignments[is_reassigned, "candidate_id"].unique()
        for affected_cand in affected_candidates:
            affected_cand_voters = new_assignments[new_assignments["candidate_id"] == affected_cand]
            if load_magnitude_method == "marginal_previous":
                marginal_utility = affected_cand_voters["utility"] - affected_cand_voters["utility_previous"]
                # defected_cand_total_load = 1 / (affected_cand_voters["utility_previous"] - old_assignments.loc[old_assignments["candidate_id"] == affected_cand, "utility_previous"]).sum()
            elif load_magnitude_method == "marginal_slate":
                is_first_assignment_voters = affected_cand_voters.loc[affected_cand_voters["2nd_selected_candidate_id"]].isna()
                marginal_utility = affected_cand_voters["utility"] 
                marginal_utility -= affected_cand_voters[~is_first_assignment_voters, "2nd_selected_candidate_id"]
                marginal_utility[is_first_assignment_voters] -= BASELINE_UTILITY
            elif load_magnitude_method == "total":
                marginal_utility = affected_cand_voters["utility"]

            # Redistribute the loads
            defected_cand_total_load = 1 / (marginal_utility.sum())
            new_assignments.loc[affected_cand_voters.index, "load"] = marginal_utility / defected_cand_total_load**2

    return new_assignments


def seq_phragmen_minimax_rated(
    rated_votes: pd.DataFrame,
    slate_size: int,
    egalitarian_utilitarian: float = 1.0,
) -> tuple[list[int], pd.Series]:
    """
    Sequential Phragmen Maximin Algorithm for rated voting.

    Adaptation of the Sequential Phragmen Minimax Voting Algorithm to rated voting.

    # Arguments
    - `rated_votes: pd.DataFrame`: Utility of each voter (rows) for each candidate (columns)
    - `slate_size: int`: The number of candidates to be selected
    - `egalitarian_utilitarian: float = 1.0`: Hyperparameter governing the egalitarian-utilitarian trade-off.

    # Returns
    - `slate: List[int]`: The slate of candidates to be selected
    - `assignments: pd.Series`: The assignments of the candidates to the voters
    """
    # TODO: Figure out egalitarian_utilitarian

    # Initialize the slate and assignments
    slate: list[int] = []
    assignments: pd.DataFrame = pd.DataFrame(
        index=rated_votes.index,
        columns=["candidate_id", "load", "utility", "utility_previous", "2nd_selected_candidate_id"],
        dtype={"candidate_id": int, "load": float, "utility": float, "utility_previous": float, "2nd_selected_candidate_id": int}
    )
    # "2nd_selected_candidate_id": the 2nd-favorite candidate for each voter among the candidates in the current slate
    
    assignments["load"] = 0
    assignments["utility_previous"] = BASELINE_UTILITY
    assignments["utility"] = BASELINE_UTILITY
    # assignments["2nd_selected_candidate_id"] = -1

    for i in range(slate_size):
        min_load = float("inf")
        min_load_candidate_id: int = -1
        for candidate in rated_votes.columns:
            assignments_with_candidate = _phragmen_update_assignments(
                assignments,
                rated_votes,
                candidate,
            )
            if assignments_with_candidate["load"].min() < min_load:
                min_load = assignments_with_candidate["load"].min()
                min_load_candidate_id = candidate
        
        slate.append(min_load_candidate_id)
        assignments = _phragmen_update_assignments(
            assignments,
            rated_votes,
            min_load_candidate_id,
        )
    
    return slate, assignments
