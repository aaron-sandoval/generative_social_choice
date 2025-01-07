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
- "marginal_previous": The load added is 1/the marginal utility created by the new candidate only among reassigned voters.
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
    reassigned = rated_votes[candidate] > new_assignments["utility"] # It matters if this is > or >= for int-valued utilities
    old_assignments = assignments.loc[reassigned]
    new_assignments.loc[reassigned, "candidate_id"] = candidate
    new_assignments.loc[reassigned, "utility"] = rated_votes[candidate]

    if load_magnitude_method == "marginal_previous":
        new_candidate_total_load: float = 1 / (rated_votes[reassigned, candidate] - old_assignments["utility"]).sum()
    elif load_magnitude_method == "marginal_slate":
        new_candidate_total_load: float = 1 / (rated_votes[reassigned, candidate] - old_assignments["utility"]).sum()
    elif load_magnitude_method == "total":
        new_candidate_total_load: float = 1 / (rated_votes[reassigned, candidate].sum())
    else:
        raise ValueError(f"Invalid load magnitude method: {load_magnitude_method}")
    
    # Calculate the loads for the reassigned voters
    new_candidate_loads: pd.Series = rated_votes[reassigned, candidate] / new_candidate_total_load**2
    if clear_reassigned_loads:
        new_assignments.loc[reassigned, "load"] = new_candidate_loads
    else:
        new_assignments.loc[reassigned, "load"] += new_candidate_loads
    
    if redistribute_defected_candidate_loads:
        affected_candidates = old_assignments["candidate_id"].unique()
        for candidate in affected_candidates:
            if load_magnitude_method == "marginal_previous":

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
        columns=["candidate_id", "load", "utility", "utility_previous"],
        dtype={"candidate_id": int, "load": float, "utility": float, "utility_previous": float}
    )
    assignments["load"] = 0
    assignments["utility_previous"] = BASELINE_UTILITY
    assignments["utility"] = BASELINE_UTILITY

    for i in range(slate_size):
        min_load = float("inf")
        min_load_candidate_id: int = -1
        for candidate in rated_votes.columns:
            aissignments_with_candidate = _phragmen_update_assignments(
                assignments,
                rated_votes,
                candidate,
            )
