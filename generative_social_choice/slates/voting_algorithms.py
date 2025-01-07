from pathlib import Path
import pandas as pd
import os
from typing import Optional
from tqdm import tqdm
import concurrent.futures

from generative_social_choice.utils.helper_functions import (
    get_time_string,
    get_base_dir_path,
)
from generative_social_choice.queries.query_interface import Agent, Generator


def phragmen_update_assignments(
    assignments: pd.DataFrame,
    rated_votes: pd.DataFrame,
    candidate: int,
) -> pd.DataFrame:
    """
    Update the assignments of the voters to the candidates.
    """
    new_assignments = assignments.copy()
    reassigned = assignments["candidate_id"] > new_assignments["utility"] # It matters if this is > or >= for int-valued utilities
    old_assignments = new_assignments.loc[reassigned].copy()
    new_assignments.loc[reassigned, "candidate_id"] = candidate
    new_assignments.loc[reassigned, "utility"] = rated_votes[candidate]


    new_assignments.loc[reassigned, "load"] += 1
    
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
        columns=["candidate_id", "load", "utility"],
        dtype={"candidate_id": int, "load": float, "utility": float}
    )

    for i in range(slate_size):
        min_load = float("inf")
        min_load_candidate_id: int = -1
        for candidate in rated_votes.columns:
            aissignments_with_candidate = phragmen_update_assignments(
                assignments,
                rated_votes,
                candidate,
            )
