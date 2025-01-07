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


def seq_phragmen_maximin(
    rated_votes: pd.DataFrame,
    slate_size: int,
    egalitarian_utilitarian: float = 1.0,
) -> tuple[list[int], pd.Series]:
    """
    Sequential Phragmen Maximin Voting Algorithm

    Adaptation of the Sequential Phragmen Maximin Voting Algorithm to rated voting.

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
    slate = []
    assignments = pd.DataFrame(
        index=rated_votes.index,
        columns=["candidate_id", "load"],
        dtype={"candidate_id": int, "load": float}
    )

    for i in range(slate_size):
        min_load = float("inf")
        min_load_candidate_id: int = -1
        for candidate in rated_votes.columns:
            candidate_load = 
