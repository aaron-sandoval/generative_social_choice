from collections import defaultdict
from functools import cached_property
from itertools import combinations
import abc
import math
from dataclasses import dataclass
from typing import Optional, Literal, override
import warnings

import pulp
import pandas as pd
import numpy as np

from generative_social_choice.slates.voting_utils import (
    voter_utilities, voter_max_utilities_from_slate, NoiseAugmentationMethod,
    CellWiseAugmentation, CandidateWiseAugmentation, VoterAndCellWiseAugmentation,
    filter_candidates_by_individual_pareto_efficiency)
from generative_social_choice.utils.helper_functions import geq_lib

class RatedVoteCase:
    """
    A voting case with rated votes and sets of possible results which satisfy various properties.

    # Arguments
    - `rated_votes: pd.DataFrame | list[list[int | float]]`: Utility of each voter (rows) for each candidate (columns)
      - If passed as a nested list, it's converted to a DataFrame with columns named `s1`, `s2`, etc.
    - `slate_size: int`: The number of candidates to be selected
    - `name: Optional[str] = None`: The name of the case
    - `noise_augmentation: bool = True`: Whether to augment the case with noise
    - `noise_augmentation_methods: dict[NoiseAugmentationMethod, int]`: The methods to use to augment the case and the number of augmentations to use for each method
    - `base_seed: Optional[int] = None`: Base seed for reproducible random noise generation. If None, uses hash of case name.
    """
    
    def __init__(
        self,
        rated_votes: pd.DataFrame | list[list[int | float]],
        slate_size: int,
        name: Optional[str] = None,
        noise_augmentation: bool = True,
        noise_augmentation_methods: dict[NoiseAugmentationMethod, int] = None,
        base_seed: Optional[int] = None
    ):
        # Set default for noise_augmentation_methods
        if noise_augmentation_methods is None:
            noise_augmentation_methods = {
                CellWiseAugmentation(): 4, 
                CandidateWiseAugmentation(): 10, 
                VoterAndCellWiseAugmentation(): 10
            }
        

        # Convert input to DataFrame and store internally
        if isinstance(rated_votes, (list, np.ndarray)):
            self._rated_votes = pd.DataFrame(rated_votes, columns=[f"s{i}" for i in range(1, len(rated_votes[0]) + 1)])
        else:
            self._rated_votes = rated_votes.copy()
        
        self.slate_size = slate_size
        self.noise_augmentation = noise_augmentation
        self.noise_augmentation_methods = noise_augmentation_methods
        
        # Generate name if not provided
        if name is None:
            cols_str = "_".join(str(col) + "_" + "_".join(str(x).replace(".", "p") for x in self._rated_votes[col]) 
                              for col in self._rated_votes.columns)
            name = f"k_{slate_size}_{cols_str}"
        self.name = name
        
        # Set base_seed using hash of name if not provided for consistent but unique seeds
        if base_seed is None:
            base_seed = hash(name) % (2**31)
        self.base_seed = base_seed

    def __str__(self) -> str:
        return f"RatedVoteCase(name={self.name}, slate_size={self.slate_size}, votes={self._rated_votes.to_numpy})"

    @property
    def rated_votes(self) -> pd.DataFrame:
        """
        Return a copy of the rated votes to prevent external mutation.
        
        This ensures that the original rated_votes DataFrame cannot be modified
        after the RatedVoteCase is constructed, maintaining immutability.
        """
        return self._rated_votes.copy()

    @cached_property
    def augmented_cases(self) -> list[pd.DataFrame]:
        """
        Return a list of rated vote matrices with noise added according to `self.noise_augmentation_methods`.
        Each augmentation method gets a unique seed derived from the base_seed for reproducible results.
        """
        if not self.noise_augmentation:
            return [self.rated_votes]
        augmented_cases = []
        method_index = 0
        for noise_method, num_augmentations in self.noise_augmentation_methods.items():
            # Create a new instance of the noise method with a derived seed
            # Use method_index and method type to ensure uniqueness
            method_seed = self.base_seed + method_index * 1000 + hash(type(noise_method).__name__) % 1000
            seeded_method = type(noise_method)(
                min_magnitude=noise_method.min_magnitude,
                max_magnitude=noise_method.max_magnitude,
                sign=noise_method.sign,
                seed=method_seed
            )
            augmented_cases.extend(seeded_method.augment(self.rated_votes, num_augmentations))
            method_index += 1
        return augmented_cases


Phragmen_Load_Magnitude = Literal["marginal_slate", "marginal_previous"]
"""
The magnitude of the load added for a new candidate.
- "marginal_slate": The load added is 1/the marginal utility created by the new candidate compared to the best alternative in the slate for each voter.
  - When reassigning, the total load of the candidate is set by the sum of marginal utilities created by that candidate compared to the best alternative in the slate for each voter.
- "marginal_previous": The load added is 1/the marginal utility created by the new candidate only among reassigned voters.
  - When reassigning, the total load of the candidate is set by the sum of marginal utilities created by that candidate compared to the candidate to which each voter was previously assigned.
- "total": REMOVED FOR POOR PERFORMANCE
  - The load added is 1/the total utility created by the new candidate among reassigned voters.
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
        Fully distinguishing name of the voting algorithm, used in labeling test cases.

        Name must be a valid Python function name.
        """
        return repr(self)
    
    @property
    def display_name(self) -> str:
        """
        Short name of the voting algorithm, used in plotting and labeling.
        """
        return self.name.replace("(", "").replace(")", "")


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
class UtilityTransformation(abc.ABC):
    """
    Abstract class for utility transformations
    """
    @abc.abstractmethod
    def transform(
        self,
        rated_votes: pd.DataFrame,
        slate_size: int
    ) -> pd.DataFrame:
        """
        Compute a vote matrix with transformed utilities

        # Arguments
        - `rated_votes: pd.DataFrame`: Utility of each voter (rows) for each candidate (columns)
        - `slate_size: int`: The number of candidates to be selected

        # Returns
        `transformed_rated_votes: pd.DataFrame`: Utility of each voter (rows) for each candidate (columns)
            after transformation
        """
        pass

    @property
    def display_name(self) -> str:
        return repr(self)

@dataclass(frozen=True)
class GeometricTransformation(UtilityTransformation):
    """
    Kolm-Pollak utility transformation with inequality aversion parameter `kappa` > 0.

    The Kolm-Pollak equally distributed equivalent (EDE) of utilities u_1, ..., u_n is
    `EDE(u) = -(1/kappa) * ln((1/n) * sum_i exp(-kappa*u_i))`.
    It is a strictly increasing function of the total transformed utility `sum_i f(u_i)` with
    `f(u) = -exp(-kappa*u)`, so maximizing that total maximizes the EDE.
    kappa -> 0 approaches utilitarian welfare (total utility) and kappa -> inf approaches egalitarian
    welfare (minimum utility).

    This was formerly parametrized by p = exp(kappa), with the normalized transformation
    `(1-(1/p)^u)/(1-(1/p))`, a positive affine function of f. The default kappa = ln(1.5) is the
    former default p=1.5.

    `transform` returns the positive affine rescaling `g(u) = -exp(kappa*(c-u))` of f, where c is the
    smallest of the voters' maximum utilities. Every algorithm using a transformation assigns each
    voter to exactly one candidate, so maximizing the total of g selects the same slate and
    assignment as maximizing the total of f. The normalized form is numerically unusable for large
    kappa: it is nearly constant, so for kappa=ln(150) and utilities of 3 or more the differences
    between candidates are ~1e-7, below the tolerances of the ILP solver. g keeps the utilities
    around the worst-off voter's best candidate of order 1. Note that g is negative.
    """
    kappa: float = math.log(1.5)

    @property
    def display_name(self) -> str:
        return f"κ={self.kappa:.3g}"

    @override
    def transform(
        self,
        rated_votes: pd.DataFrame,
        slate_size: int
    ) -> pd.DataFrame:
        """
        Compute a vote matrix with transformed utilities

        # Arguments
        - `rated_votes: pd.DataFrame`: Utility of each voter (rows) for each candidate (columns)
        - `slate_size: int`: The number of candidates to be selected

        # Returns
        `transformed_rated_votes: pd.DataFrame`: Utility of each voter (rows) for each candidate (columns)
            after transformation. These are the rescaled utilities g(u) = -exp(kappa*(c-u)) described in
            the class docstring, an equivalent objective to the Kolm-Pollak EDE.
        """
        anchor = rated_votes.max(axis=1).min()
        return -np.exp(self.kappa * (anchor - rated_votes))

@dataclass(frozen=True)
class SequentialPhragmenMinimax(VotingAlgorithm):
    load_magnitude_method: Phragmen_Load_Magnitude = "marginal_slate"
    clear_reassigned_loads: bool = False
    redistribute_defected_candidate_loads: bool = False

    @property
    def name(self) -> str:
        return f"Phragmen({self.load_magnitude_method}, clear={self.clear_reassigned_loads}, redist={self.redistribute_defected_candidate_loads})"

    @property
    def display_name(self) -> str:
        return f"Phragmen"
        # return f"Phragmen({self.load_magnitude_method},{"T" if self.clear_reassigned_loads else "F"},{"T" if self.redistribute_defected_candidate_loads else "F"})"

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
        rated_votes = filter_candidates_by_individual_pareto_efficiency(rated_votes)

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
        valid_candidates = defaultdict(lambda: None)
        for candidate in rated_votes.loc[:, :NULL_CANDIDATE_ID].columns:
            valid_candidates[candidate] = None
        valid_candidates.pop(NULL_CANDIDATE_ID)
        # valid_candidates = set(rated_votes.loc[:, :NULL_CANDIDATE_ID].columns) - rejected_candidates

        while len(slate) < slate_size and i <= len(rated_votes.columns)+1:
            i += 1
            min_load = float("inf")
            min_load_among_reassigned_voters = float("inf")
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
                reassigned_voters = assignments_with_candidate["candidate_id"] != assignments["candidate_id"]
                if not reassigned_voters.any():
                    continue
                assert (assignments_with_candidate.loc[reassigned_voters].candidate_id == candidate).all()
                all_voters_max_load = assignments_with_candidate["load"].max()
                reassigned_max_load = assignments_with_candidate.loc[reassigned_voters, "load"].max()


                if not geq_lib((all_voters_max_load, reassigned_max_load), (min_load, min_load_among_reassigned_voters), abs_tol=1e-9):
                    assert reassigned_voters.any()
                    min_load = all_voters_max_load
                    min_load_among_reassigned_voters = reassigned_max_load
                    min_load_candidate_id = candidate
                    min_load_assignments = assignments_with_candidate.copy()

            if min_load_candidate_id == NULL_CANDIDATE_ID:
                warnings.warn(f"No improvements possible beyond the partial length-{len(slate)}/{slate_size} slate: {slate}. Returning the partial slate.")
                break
            slate.append(min_load_candidate_id)
            assignments = min_load_assignments
            # assert (assignments.loc[reassigned_voters].candidate_id == min_load_candidate_id).all()
            valid_candidates.pop(min_load_candidate_id)
                
        # Remove the null candidate column in case the modification would persist outside the function
        rated_votes = rated_votes.drop(columns=[NULL_CANDIDATE_ID])

        # Check that the assignments are valid and assign any NULL_CANDIDATE_ID entries to their max utility candidate from the slate
        null_assigned = assignments["candidate_id"] == NULL_CANDIDATE_ID
        if any(null_assigned):
            max_utility_assignments = voter_max_utilities_from_slate(rated_votes, slate)
            assignments.loc[null_assigned, "candidate_id"] = max_utility_assignments.loc[
                null_assigned, "candidate_id"]
            assignments.loc[null_assigned, "utility"] = max_utility_assignments.loc[
                null_assigned, "utility"]

        return slate, assignments

    def _phragmen_update_assignments(
        self,
        assignments: pd.DataFrame,
        rated_votes: pd.DataFrame,
        candidate: str,
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
        # print(f"rated_votes=\n{rated_votes}")
        # print(f"rated_votes[candidate]=\n{rated_votes[candidate]}")
        # print(f"new_assignments['utility']=\n{new_assignments['utility']}")
        is_reassigned = rated_votes[candidate] > new_assignments["utility"] # It matters if this is > or >= for int-valued utilities
        reassignments = assignments.loc[is_reassigned]
        old_assignments = assignments.loc[is_reassigned]
        new_assignments.loc[is_reassigned, "second_selected_candidate_id"] = reassignments["candidate_id"]
        new_assignments.loc[is_reassigned, "candidate_id"] = candidate
        new_assignments.loc[is_reassigned, "utility_previous"] = reassignments["utility"] # TODO: Check that this makes a deep copy
        new_assignments.loc[is_reassigned, "utility"] = rated_votes[candidate]

        if any(is_reassigned):
            if self.load_magnitude_method == "marginal_previous":
                new_candidate_total_load: float = 1 / (rated_votes.loc[is_reassigned, candidate] - old_assignments["utility"]).sum()
            elif self.load_magnitude_method == "marginal_slate":
                # Convert to numpy view if possible (avoids O(nm) copy when DataFrame is contiguous)
                # Safe to use view since we only read from it, never modify
                rated_votes_np = rated_votes.to_numpy(copy=False)
                
                # is_first_assignment_voters = new_assignments.loc[is_reassigned, "second_selected_candidate_id"].isna()
                marginal_utility = new_assignments.loc[is_reassigned, "utility"]
                # Optimized: O(n) lookup instead of O(n*m) DataFrame creation + diagonal extraction
                second_cand_ids = new_assignments.loc[is_reassigned, "second_selected_candidate_id"]
                row_indices = rated_votes.index.get_indexer(reassignments.index)
                col_indices = rated_votes.columns.get_indexer(second_cand_ids)
                # Raise exception if any indices are invalid (get_indexer returns -1 for missing)
                if np.any(row_indices < 0):
                    invalid_rows = reassignments.index[row_indices < 0]
                    raise ValueError(
                        f"Invalid row indices found: {invalid_rows.tolist()}. "
                        f"These voters are not in rated_votes.index."
                    )
                if np.any(col_indices < 0):
                    invalid_cols = second_cand_ids[col_indices < 0]
                    raise ValueError(
                        f"Invalid column indices found: {invalid_cols.tolist()}. "
                        f"These candidate IDs are not in rated_votes.columns."
                    )
                next_best_utility = rated_votes_np[row_indices, col_indices]
                marginal_utility -= pd.Series(next_best_utility, index=reassignments.index, dtype=float).fillna(BASELINE_UTILITY)
                new_candidate_total_load: float = 1 / (marginal_utility.sum())

                # Update 2nd-favorite candidate for each voter
                # Optimized: O(n) lookup instead of O(n*m) reindex + numpy conversion
                rated_votes["second_fav_cand_id"] = new_assignments.second_selected_candidate_id
                row_indices_all = np.arange(len(rated_votes))
                col_indices_all = rated_votes.columns.get_indexer(rated_votes["second_fav_cand_id"])
                # Raise exception if any indices are invalid (get_indexer returns -1 for missing)
                if np.any(col_indices_all < 0):
                    invalid_cols = rated_votes["second_fav_cand_id"][col_indices_all < 0]
                    raise ValueError(
                        f"Invalid column indices found: {invalid_cols.tolist()}. "
                        f"These candidate IDs are not in rated_votes.columns."
                    )
                cur_2nd_fav_utility = pd.Series(
                    rated_votes_np[row_indices_all, col_indices_all],
                    index=rated_votes.index
                )
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
                        next_best_utility = voter_utilities(rated_votes, affected_cand_voters["second_selected_candidate_id"]).values
                        # next_best_utility = np.diag(rated_votes.loc[affected_cand_voters.index, affected_cand_voters["second_selected_candidate_id"]].to_numpy())
                        marginal_utility -= pd.Series(next_best_utility, index=affected_cand_voters.index).fillna(BASELINE_UTILITY)
                        # marginal_utility[is_first_assignment_voters] -= BASELINE_UTILITY
                    elif self.load_magnitude_method == "total":
                        marginal_utility = affected_cand_voters["utility"]

                    # Redistribute the loads
                    defected_cand_total_load = 1 / (marginal_utility.sum())
                    new_assignments.loc[affected_cand_voters.index, "load"] = marginal_utility * defected_cand_total_load**2

        return new_assignments

@dataclass(frozen=True)
class ExactTotalUtilityMaximization(VotingAlgorithm):
    # name = "ExactTotalUtilityMaximization"
    utility_transform: Optional[UtilityTransformation] = None

    @property
    def display_name(self) -> str:
        return f"U-CC-Exact{f'({self.utility_transform.display_name})' if self.utility_transform is not None else ''}"
        
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
            - `utility`: The utility of the voter for the candidate
        """
        if self.utility_transform is not None:
            rated_votes = self.utility_transform.transform(rated_votes=rated_votes, slate_size=slate_size)
        
        # First we formulate the problem as integer programming problem
        num_agents = rated_votes.shape[0]
        num_statements = rated_votes.shape[1]
        U = rated_votes.to_numpy()  # Utility matrix

        # Initialize the problem
        prob = pulp.LpProblem("Maximize_Utility", pulp.LpMaximize)

        # Decision variables
        x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(num_statements)]  # Whether to select statement j
        y = [
            [pulp.LpVariable(f"y_{i}_{j}", cat="Binary") for j in range(num_statements)]  # Agent i assigned to statement j
            for i in range(num_agents)
        ]

        # Objective function: Maximize total utility
        prob += pulp.lpSum(U[i][j] * y[i][j] for i in range(num_agents) for j in range(num_statements)), "TotalUtility"

        # Constraint 1: Select exactly k statements
        prob += pulp.lpSum(x[j] for j in range(num_statements)) == slate_size, "Select_k_Statements"

        # Constraint 2: Each agent is assigned to exactly one statement
        for i in range(num_agents):
            prob += pulp.lpSum(y[i][j] for j in range(num_statements)) == 1, f"Assign_Agent_{i}"

        # Constraint 3: An agent can only be assigned to a selected statement
        for i in range(num_agents):
            for j in range(num_statements):
                prob += y[i][j] <= x[j], f"Assign_Agent_{i}_to_Selected_Statement_{j}"

        # Now solve the problem using an existing integer programming solver
        prob.solve(pulp.PULP_CBC_CMD(msg=0, gapRel=1e-9, gapAbs=1e-12))

        # Extract slate and assignments from the solved problem
        slate: list[str] = []
        for j in range(num_statements):
            if pulp.value(x[j]) > 0.5:  # Selected statement
                slate.append(rated_votes.columns[j])

        candidate_assignments: list[str] = []
        for i in range(num_agents):
            for j in range(num_statements):
                if pulp.value(y[i][j]) > 0.5:  # Assignment made
                    candidate_assignments.append(rated_votes.columns[j])
        assignments: pd.DataFrame = pd.DataFrame(index=rated_votes.index)
        assignments["candidate_id"] = pd.Series(candidate_assignments, index=assignments.index, dtype=str)

        return slate, assignments


EgalitarianTiebreak = Literal["leximin", "utilitarian", "none"]
"""
How the exact egalitarian algorithms choose among the solutions achieving the optimal minimum utility.
Many solutions typically achieve it, and the solver would return an arbitrary one of them.
- "leximin": Maximize the second-lowest utility among those solutions, then the third-lowest, and so on.
- "utilitarian": Maximize total utility among those solutions.
- "none": Return the solver's arbitrary choice.
"""
_TIEBREAK_DISPLAY_SUFFIX: dict[str, str] = {"leximin": "", "utilitarian": "(utilitarian tiebreak)", "none": "(no tiebreak)"}


def _exact_egalitarian_solution(
    rated_votes: pd.DataFrame,
    slate_size: int,
    coalition_size_bounds: Optional[tuple[int, int]],
    tiebreak: EgalitarianTiebreak,
) -> tuple[list[str], list[str]]:
    """
    Jointly select a slate and assign each voter to one of its members, maximizing the minimum utility
    and breaking ties according to `tiebreak`. Solved exactly as a sequence of integer programs.

    # Arguments
    - `coalition_size_bounds: Optional[tuple[int, int]]`: The minimum and maximum number of voters assigned to
      each slate member (Monroe), or None for any number of voters (CC). Under CC each voter is assigned to their
      favorite slate member.
    - `tiebreak: EgalitarianTiebreak`: How to choose among the solutions achieving the optimal minimum utility

    # Leximin
    For an assignment, let N(t) be the number of voters with utility <= t, and N<(t) the number with utility < t.
    The sorted utility vector of one assignment is leximin-better than that of another iff its N is smaller at
    the lowest threshold where the two differ. So leximin is built up level by level. Given c_prev voters at or
    below the levels found so far:
    1. The next level t is the largest achievable (c_prev+1)-th lowest utility.
    2. c is the smallest achievable number of voters with utility <= t. It is at least c_prev+1, and only needs
       a second integer program if the solution of step 1 has more.
    Each level adds the constraints N<(t) <= c_prev and N(t) <= c, which are linear in the assignment variables
    and hold for every leximin-optimal solution. The levels are read from the data rather than from the
    solver's objective values, which avoids losing ties to floating point error.

    Once no other slate satisfies the constraints, the slate is leximin-optimal. Under CC, the favorite
    assignment is then leximin-optimal too, so the search stops. Under Monroe, the remaining levels only
    determine the assignment, so the search continues on the ballot profile restricted to the slate. It stops
    once every voter is at or below some level (c = n).

    # Returns
    - `slate: list[str]`: The selected candidates
    - `candidate_assignments: list[str]`: The candidate assigned to each voter, in the order of `rated_votes.index`
    """
    num_agents, num_statements = rated_votes.shape
    U = rated_votes.to_numpy()  # Utility matrix
    big_m = float(U.max() - U.min())  # Upper bound on the difference between any two utilities

    # (level, strict, cap): at most `cap` agents have utility < level (strict) or <= level (not strict)
    count_constraints: list[tuple[float, bool, int]] = []

    def solve(
        objective: Literal["kth_lowest", "count", "total", "feasibility"],
        columns: list[int],
        rank: int = 0,
        threshold: Optional[float] = None,
        excluded_slate: Optional[list[int]] = None,
    ) -> Optional[tuple[list[int], np.ndarray]]:
        """
        Solve one integer program over the statements `columns`, subject to `count_constraints`.

        Objectives:
        - "kth_lowest": Maximize the (rank+1)-th lowest utility. rank=0 is the minimum utility.
        - "count": Minimize the number of agents with utility <= `threshold`
        - "total": Maximize total utility
        - "feasibility": Find any solution whose slate differs from `excluded_slate`

        Returns the selected statements and the statement assigned to each agent, as indices into the columns of
        `rated_votes`, or None if a "feasibility" problem is infeasible.
        """
        num_cols = len(columns)
        U_sub = U[:, columns]
        prob = pulp.LpProblem(
            f"Egalitarian_{objective}", pulp.LpMinimize if objective == "count" else pulp.LpMaximize
        )

        # Decision variables
        x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(num_cols)]  # Whether to select statement j
        y = [
            [pulp.LpVariable(f"y_{i}_{j}", cat="Binary") for j in range(num_cols)]  # Agent i assigned to statement j
            for i in range(num_agents)
        ]
        agent_utility = [pulp.lpSum(U_sub[i][j] * y[i][j] for j in range(num_cols)) for i in range(num_agents)]

        if objective == "kth_lowest":
            z = pulp.LpVariable("z", cat="Continuous")  # Lower bound on the utility of all but `rank` agents
            prob += z, "KthLowestUtility"
            if rank == 0:
                for i in range(num_agents):
                    prob += z <= agent_utility[i], f"Minimum_Utility_Bound_{i}"
            else:
                # w_i = 1 exempts agent i from the bound. At most `rank` agents are exempt.
                w = [pulp.LpVariable(f"w_{i}", cat="Binary") for i in range(num_agents)]
                for i in range(num_agents):
                    prob += z <= agent_utility[i] + big_m * w[i], f"Kth_Lowest_Utility_Bound_{i}"
                prob += pulp.lpSum(w) <= rank, "Exempt_Agents"
        elif objective == "count":
            prob += pulp.lpSum(
                y[i][j] for i in range(num_agents) for j in range(num_cols) if U_sub[i][j] <= threshold
            ), "Agents_At_Or_Below_Threshold"
        elif objective == "total":
            prob += pulp.lpSum(agent_utility), "TotalUtility"
        else:
            prob += pulp.lpSum(x), "Constant"  # Fixed by the slate size constraint

        # Select exactly k statements
        prob += pulp.lpSum(x) == slate_size, "Select_k_Statements"

        # Each agent is assigned to exactly one statement
        for i in range(num_agents):
            prob += pulp.lpSum(y[i]) == 1, f"Assign_Agent_{i}"

        # An agent can only be assigned to a selected statement
        for i in range(num_agents):
            for j in range(num_cols):
                prob += y[i][j] <= x[j], f"Assign_Agent_{i}_to_Selected_Statement_{j}"

        # Monroe: each selected statement gets floor(n/k) or ceil(n/k) agents, and unselected statements get none.
        if coalition_size_bounds is not None:
            min_coalition_size, max_coalition_size = coalition_size_bounds
            for j in range(num_cols):
                coalition_size = pulp.lpSum(y[i][j] for i in range(num_agents))
                prob += coalition_size >= min_coalition_size * x[j], f"Min_Coalition_Size_{j}"
                prob += coalition_size <= max_coalition_size * x[j], f"Max_Coalition_Size_{j}"

        # Bound the number of agents below each level. Comparing against the data directly, rather than
        # constraining the solver's continuous utility sums, avoids losing ties to floating point error.
        for c, (level, strict, cap) in enumerate(count_constraints):
            below = [
                y[i][j] for i in range(num_agents) for j in range(num_cols)
                if (U_sub[i][j] < level if strict else U_sub[i][j] <= level)
            ]
            if cap == 0:
                for var in below:
                    var.upBound = 0
            elif below and cap < num_agents:
                prob += pulp.lpSum(below) <= cap, f"Count_{c}"

        if excluded_slate is not None:
            prob += pulp.lpSum(x[columns.index(j)] for j in excluded_slate) <= slate_size - 1, "Exclude_Slate"

        prob.solve(pulp.PULP_CBC_CMD(msg=0, gapRel=1e-9, gapAbs=1e-12))
        status = pulp.LpStatus[prob.status]
        if objective == "feasibility" and status == "Infeasible":
            return None
        if status != "Optimal":
            raise RuntimeError(f"{prob.name} not solved to optimality: {status}")

        slate = [columns[j] for j in range(num_cols) if pulp.value(x[j]) > 0.5]
        if coalition_size_bounds is None:
            # Assign each agent to their favorite statement in the slate. This weakly raises every agent's
            # utility over the solver's assignment, so it is still optimal and satisfies the count constraints.
            assigned = np.array(slate)[U[:, slate].argmax(axis=1)]
        else:
            assigned = np.array([
                columns[next(j for j in range(num_cols) if pulp.value(y[i][j]) > 0.5)] for i in range(num_agents)
            ])
        return slate, assigned

    def agent_utilities(solution: tuple[list[int], np.ndarray]) -> np.ndarray:
        return U[np.arange(num_agents), solution[1]]

    columns = list(range(num_statements))
    solution = solve("kth_lowest", columns)

    if tiebreak == "utilitarian":
        # Forbid assignments below the optimal minimum utility, read from the data rather than the solver's
        # objective value
        count_constraints.append((agent_utilities(solution).min(), True, 0))
        solution = solve("total", columns)
    elif tiebreak == "leximin":
        num_settled = 0  # c_prev: the number of agents with utility at or below the levels found so far
        slate_is_unique = False
        while True:
            sorted_utilities = np.sort(agent_utilities(solution))
            level = sorted_utilities[num_settled]
            count_constraints.append((level, True, num_settled))
            num_at_or_below = int((sorted_utilities <= level).sum())
            if num_at_or_below > num_settled + 1:
                solution = solve("count", columns, threshold=level)
                num_at_or_below = int((agent_utilities(solution) <= level).sum())
            count_constraints.append((level, False, num_at_or_below))
            num_settled = num_at_or_below
            if num_settled == num_agents:
                break

            if not slate_is_unique:
                slate_is_unique = solve("feasibility", columns, excluded_slate=solution[0]) is None
                if slate_is_unique:
                    if coalition_size_bounds is None:
                        break
                    columns = solution[0]
            solution = solve("kth_lowest", columns, rank=num_settled)
    elif tiebreak != "none":
        raise ValueError(f"Invalid tiebreak: {tiebreak}")

    slate_ix, assigned_ix = solution
    return [rated_votes.columns[j] for j in slate_ix], [rated_votes.columns[j] for j in assigned_ix]


@dataclass(frozen=True)
class ExactMinimumUtilityMaximization(VotingAlgorithm):
    """
    Egalitarian Chamberlin-Courant: select the slate maximizing the minimum utility of any voter,
    where each voter is assigned to their favorite candidate in the slate.

    Solved exactly as a sequence of integer programs (see `_exact_egalitarian_solution`). Many slates typically
    achieve the optimal minimum utility, and the solver would return an arbitrary one of them, so ties are broken
    according to `tiebreak`.

    # Arguments
    - `tiebreak: EgalitarianTiebreak = "leximin"`: How to break ties in minimum utility
    """
    tiebreak: EgalitarianTiebreak = "leximin"

    @property
    def display_name(self) -> str:
        return "E-CC-Exact" + _TIEBREAK_DISPLAY_SUFFIX[self.tiebreak]

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
            - `utility`: The utility of the voter for the candidate
        """
        slate, _ = _exact_egalitarian_solution(rated_votes, slate_size, None, self.tiebreak)

        # Assign each agent to their favorite statement in the slate
        assignments: pd.DataFrame = pd.DataFrame(index=rated_votes.index)
        assignments["candidate_id"] = pd.Series(rated_votes.loc[:, slate].idxmax(axis=1), index=assignments.index, dtype=str)
        assignments["utility"] = rated_votes.loc[:, slate].max(axis=1)

        return slate, assignments


@dataclass(frozen=True)
class ExactTotalUtilityMaximizationBruteSearch(VotingAlgorithm):
    """Brute-force search over all slates of size k to maximize total utility."""

    utility_transform: Optional[UtilityTransformation] = None

    @property
    def display_name(self) -> str:
        return f"U-CC-ExactBrute{f'({self.utility_transform.display_name})' if self.utility_transform is not None else ''}"

    @override
    def vote(
        self,
        rated_votes: pd.DataFrame,
        slate_size: int,
    ) -> tuple[list[str], pd.DataFrame]:
        """
        Enumerate all slates of size `slate_size` and choose the one with the
        highest total utility. Voters are assigned to their best candidate
        within the chosen slate.

        # Returns
        - `slate: List[str]`: The slate of candidates to be selected
        - `assignments: pd.DataFrame`: The assignments of the candidates to the
          voters with at least the column:
            - `candidate_id`: The candidate to which the voter is assigned
            - `utility`: The utility of the voter for the assigned candidate
        """
        if slate_size > len(rated_votes.columns):
            raise ValueError(
                "slate_size cannot exceed the number of available candidates."
            )

        if self.utility_transform is not None:
            rated_votes = self.utility_transform.transform(
                rated_votes=rated_votes, slate_size=slate_size
            )

        best_slate: Optional[tuple[str, ...]] = None
        best_total_utility: float = -np.inf

        for slate_candidates in combinations(rated_votes.columns, slate_size):
            slate_votes = rated_votes.loc[:, slate_candidates]
            utilities = slate_votes.max(axis=1)
            total_utility = utilities.sum()

            if total_utility > best_total_utility:
                best_total_utility = total_utility
                best_slate = slate_candidates

        assert best_slate is not None
        slate_votes = rated_votes.loc[:, best_slate]
        best_utilities = slate_votes.max(axis=1)
        best_candidate_id = slate_votes.idxmax(axis=1)

        assignments = pd.DataFrame(index=rated_votes.index)
        assignments["candidate_id"] = pd.Series(
            best_candidate_id, index=assignments.index, dtype=str
        )
        assignments["utility"] = pd.Series(
            best_utilities, index=assignments.index, dtype=float
        )

        return list(best_slate), assignments

@dataclass(frozen=True)
class LPTotalUtilityMaximization(VotingAlgorithm):
    """Linear programming relaxation of the integer programming problem

    For this approach, we can't guarantee finding an optimal solution but the algorithm runs
    in polynomial time."""
    # name = "LPTotalUtilityMaximization"
    utility_transform: Optional[UtilityTransformation] = None

    
    @property
    def display_name(self) -> str:
        return f"U-CC-LP{f'({self.utility_transform.display_name})' if self.utility_transform is not None else ''}"

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
            - `utility`: The utility of the voter for the candidate
        """
        if self.utility_transform is not None:
            rated_votes = self.utility_transform.transform(rated_votes=rated_votes, slate_size=slate_size)
        
        # First we formulate the problem as integer programming problem
        num_agents = rated_votes.shape[0]
        num_statements = rated_votes.shape[1]
        U = rated_votes.to_numpy()  # Utility matrix

        # Initialize the problem
        prob = pulp.LpProblem("Maximize_Utility_LP_Relaxation", pulp.LpMaximize)

        # Decision variables (now relaxed to continuous)
        x = [pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat="Continuous") for j in range(num_statements)]
        y = [
            [pulp.LpVariable(f"y_{i}_{j}", lowBound=0, upBound=1, cat="Continuous") for j in range(num_statements)]
            for i in range(num_agents)
        ]

        # Objective function: Maximize total utility
        prob += pulp.lpSum(U[i][j] * y[i][j] for i in range(num_agents) for j in range(num_statements)), "TotalUtility"

        # Constraint 1: Select exactly k statements
        prob += pulp.lpSum(x[j] for j in range(num_statements)) == slate_size, "Select_k_Statements"

        # Constraint 2: Each agent is assigned to exactly one statement
        for i in range(num_agents):
            prob += pulp.lpSum(y[i][j] for j in range(num_statements)) == 1, f"Assign_Agent_{i}"

        # Constraint 3: An agent can only be assigned to a selected statement
        for i in range(num_agents):
            for j in range(num_statements):
                prob += y[i][j] <= x[j], f"Assign_Agent_{i}_to_Selected_Statement_{j}"

        # Now solve the problem using an existing integer programming solver
        prob.solve(pulp.PULP_CBC_CMD(msg=0, gapRel=1e-9, gapAbs=1e-9))

        # Extract slate and assignments from the solved problem
        # Here, take the k statements with maximum score
        #TODO Make it possible to do this based on stochastic selection as well?
        statemend_ixs = np.array([pulp.value(x[j]) for j in range(num_statements)]).argsort()[::-1][:slate_size]
        slate: list[str] = [rated_votes.columns[j] for j in statemend_ixs]

        # We assign each agent to the statement with highest utility
        assignments: pd.DataFrame = pd.DataFrame(index=rated_votes.index)
        assignments["candidate_id"] = pd.Series(rated_votes.loc[:, slate].idxmax(axis=1), index=assignments.index, dtype=str)

        return slate, assignments

@dataclass(frozen=True)
class GreedyTotalUtilityMaximization(VotingAlgorithm):
    # name = "GreedyTotalUtilityMaximization"
    utility_transform: Optional[UtilityTransformation] = None

    
    @property
    def display_name(self) -> str:
        return f"U-CC-Greedy{f'({self.utility_transform.display_name})' if self.utility_transform is not None else ''}"

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
            - `utility`: The utility of the voter for the candidate
        """
        if self.utility_transform is not None:
            rated_votes = self.utility_transform.transform(rated_votes=rated_votes, slate_size=slate_size)
        
        # Initialize the slate and assignments
        slate: list[str] = []
        assignments: pd.DataFrame = pd.DataFrame(index=rated_votes.index)
        cols={"candidate_id": str, "utility": float}
        for col, dtype in cols.items():
            assignments[col] = pd.Series(index=assignments.index, dtype=dtype)

        assignments["utility"] = BASELINE_UTILITY
        assignments["candidate_id"] = NULL_CANDIDATE_ID

        for i in range(slate_size):
            current_total_utility = assignments["utility"].sum()

            # For every candidate c not yet included in the slate, compute the maximum obtainable total utility
            # for any mapping corresponding to the slate after adding c
            best_candidate = None
            # The first candidate is always added: utilities may be negative (e.g., after
            # GeometricTransformation), so they need not improve on BASELINE_UTILITY.
            new_total_utility = current_total_utility if slate else -np.inf
            for c in rated_votes.columns:
                if c in slate:
                    continue

                combined_slate = slate + [c]

                # Compare total utilities
                potential_total_utility = rated_votes.loc[:, combined_slate].max(axis=1).sum()
                if potential_total_utility>new_total_utility:
                    best_candidate = c
                    new_total_utility = potential_total_utility

            # Stop if no further improvement is possible
            if best_candidate is None or (slate and new_total_utility<=current_total_utility):
                break

            # Update the slate and assignments
            slate.append(best_candidate)
            assignments["candidate_id"] = rated_votes.loc[:, slate].idxmax(axis=1)
            assignments["utility"] = rated_votes.loc[:, slate].max(axis=1)

        #TODO If a utility transformation was applied, assignments["utilities"] won't match the original utilities
        return slate, assignments


@dataclass(frozen=True)
class StepwiseMonroe(VotingAlgorithm):
    """
    Greedy stepwise Monroe rule applied to a complete, precomputed utility matrix.

    This is a port of the greedy selection loop in
    `slate_generation.generate_slate_ensemble_greedy`, which normally interleaves on-demand
    statement generation and utility evaluation (querying only the currently-unmatched agents each
    round) with slate selection. Here the complete utility matrix is available from the start, so
    each round reads the relevant utilities directly instead of generating/evaluating them.

    Each round selects one slate member: each candidate is scored over its best possible coalition,
    the `coalition_size` still-unmatched agents with the highest utility for it (see `objective`).
    The highest-scoring candidate is added to the slate, and that coalition is matched to it.

    # Arguments
    - `objective: Literal["minimum", "total"] = "minimum"`: How a candidate's coalition is scored each round.
      - `"minimum"` (E-Monroe-Greedy): the minimum utility in the coalition, i.e., the
        `coalition_size`-th highest utility among the unmatched agents.
      - `"total"` (U-Monroe-Greedy): the total utility of the coalition, i.e., the sum of the
        `coalition_size` highest utilities among the unmatched agents.
    - `final_assignment: Literal["greedy_equal", "free", "optimal_equal"] = "greedy_equal"`:
      Controls only the final voter->candidate assignment; the slate selection above is identical
      for all three modes.
      - `"greedy_equal"` (default, true Monroe): each voter is assigned to the candidate of the
        equal-sized coalition (~n/k voters) it was matched into during the greedy selection.
      - `"free"` (relaxed): the equal-coalition-size constraint is dropped and each voter is
        reassigned to its favorite member of the final slate (the same final assignment rule the
        utility-maximizing algorithms use).
      - `"optimal_equal"`: the equal-coalition-size constraint is *kept*, but instead of the
        greedy per-round matching the assignment is recomputed optimally over the final slate via
        `optimize_monroe_matching` -- an ILP that maximizes total utility subject to every slate
        member receiving floor(n/k) or ceil(n/k) voters (exactly n/k when k divides n).
    """
    final_assignment: Literal["greedy_equal", "free", "optimal_equal"] = "greedy_equal"
    objective: Literal["minimum", "total"] = "minimum"

    @property
    def name(self) -> str:
        # The default objective is omitted so names of the original (egalitarian) variants are unchanged
        objective_str = f", objective={self.objective}" if self.objective != "minimum" else ""
        return f"StepwiseMonroe(final_assignment={self.final_assignment}{objective_str})"

    @property
    def display_name(self) -> str:
        prefix = "E" if self.objective == "minimum" else "U"
        return prefix + {
            "greedy_equal": "-Monroe-Greedy",
            "free": "-Monroe-Greedy(CC reassign)",
            "optimal_equal": "-Monroe-Greedy(reassign)",
        }[self.final_assignment]

    @override
    def vote(
        self,
        rated_votes: pd.DataFrame,
        slate_size: int,
    ) -> tuple[list[str], pd.DataFrame]:
        """
        # Returns
        - `slate: List[str]`: The slate of candidates to be selected
        - `assignments: pd.DataFrame`: The assignments of the candidates to the voters with columns:
            - `candidate_id`: GUARANTEED: The candidate to which the voter is assigned
            - `utility`: The utility of the voter for the assigned candidate
        """
        n = len(rated_votes.index)
        k = slate_size
        voters = list(rated_votes.index)

        assignments: pd.DataFrame = pd.DataFrame(index=rated_votes.index)
        assignments["candidate_id"] = NULL_CANDIDATE_ID
        assignments["utility"] = float(BASELINE_UTILITY)

        matched: dict = {v: None for v in voters}
        slate: list[str] = []

        for round_num in range(k):
            unmatched = [v for v in voters if matched[v] is None]
            if not unmatched:
                break

            # Coalition size: identical formula to generate_slate_ensemble_greedy, ensuring that the
            # per-round coalition sizes sum to exactly n over the k rounds.
            j = round_num + 1
            coalition_size = math.ceil(n / k) if j <= n - k * (n // k) else n // k
            coalition_size = min(coalition_size, len(unmatched))

            remaining_candidates = [c for c in rated_votes.columns if c not in slate]

            # Score each remaining candidate over the coalition_size highest utilities among the
            # unmatched agents: their minimum (the coalition_size-th highest) or their sum.
            # Selection via np.partition keeps each round O(mn), giving O(kmn) overall; a full sort
            # would add an unnecessary log factor.
            sub = rated_votes.loc[unmatched, remaining_candidates].to_numpy()
            top_utils = -np.partition(-sub, coalition_size - 1, axis=0)[:coalition_size, :]
            if self.objective == "minimum":
                scores = top_utils[coalition_size - 1, :]
            else:
                scores = top_utils.sum(axis=0)
            best = remaining_candidates[int(np.argmax(scores))]
            slate.append(best)

            # Match the top coalition_size unmatched agents (by utility for `best`) to `best`.
            best_utils = rated_votes.loc[unmatched, best]
            top_idx = np.argpartition(-best_utils.to_numpy(), coalition_size - 1)[:coalition_size]
            top_voters = best_utils.index[top_idx]
            assignments.loc[top_voters, "candidate_id"] = best
            assignments.loc[top_voters, "utility"] = rated_votes.loc[top_voters, best].to_numpy()
            for v in top_voters:
                matched[v] = best

        if self.final_assignment != "greedy_equal":
            assignments = assign_to_fixed_slate(rated_votes, slate, self.final_assignment)

        return slate, assignments


def assign_to_fixed_slate(
    rated_votes: pd.DataFrame,
    slate: list[str],
    mode: Literal["free", "optimal_equal"],
) -> pd.DataFrame:
    """
    Assign every voter to a member of an already-selected slate.

    Only the `slate` columns of `rated_votes` are read, so this can also be applied to a
    utility matrix which is complete only on the slate columns.

    # Arguments
    - `mode`:
      - `"free"`: each voter is assigned to its favorite slate member (CC).
      - `"optimal_equal"`: the assignment maximizes total utility subject to every slate member
        receiving floor(n/k) or ceil(n/k) voters, via the ILP `optimize_monroe_matching`.

    # Returns
    - `assignments: pd.DataFrame` indexed like `rated_votes`, with columns `candidate_id` and `utility`
    """
    assignments: pd.DataFrame = pd.DataFrame(index=rated_votes.index)
    if mode == "free":
        # Drop the equal-coalition assignment; assign each voter to its favorite slate member.
        best_in_slate = voter_max_utilities_from_slate(rated_votes, slate)
        assignments["candidate_id"] = best_in_slate["candidate_id"]
        assignments["utility"] = best_in_slate["utility"]
    elif mode == "optimal_equal":
        # Keep equal-sized coalitions, but recompute the assignment optimally over the final
        # slate: an ILP maximizing total utility subject to each slate member getting exactly
        # n/k voters. Imported lazily so this module doesn't hard-depend on gurobipy.
        from generative_social_choice.paper_replication.compute_matching import (
            optimize_monroe_matching,
        )

        voters = list(rated_votes.index)
        util_sub = rated_votes.loc[voters, slate]
        matching_ix = optimize_monroe_matching(util_sub.to_numpy().tolist())
        assigned = pd.Series([slate[ix] for ix in matching_ix], index=voters)
        assignments["candidate_id"] = assigned
        assignments["utility"] = pd.Series(
            [rated_votes.loc[v, c] for v, c in zip(voters, assigned)], index=voters
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return assignments


@dataclass(frozen=True)
class ExactMonroe(VotingAlgorithm):
    """
    Exact Monroe rule: jointly select the slate and an assignment of equal-sized constituencies of voters
    to its members, maximizing total or minimum utility.

    Each slate member is assigned floor(n/k) or ceil(n/k) voters (exactly n/k when k divides n). Unlike
    `optimize_monroe_matching`, which finds the optimal such assignment for a fixed slate, this optimizes
    the slate too, so with `objective="total"` it is the exact counterpart of
    `StepwiseMonroe(final_assignment="optimal_equal")`. Solved as an integer program.

    # Arguments
    - `objective: Literal["total", "minimum"] = "total"`: Utility to maximize over the voters' assigned utilities.
      - `"total"`: Utilitarian Monroe
      - `"minimum"`: Egalitarian Monroe. Many solutions typically achieve the optimal minimum utility, and the
        solver would return an arbitrary one of them, so ties are broken according to `tiebreak`. Solved as a
        sequence of integer programs (see `_exact_egalitarian_solution`).
    - `tiebreak: EgalitarianTiebreak = "leximin"`: How to break ties in minimum utility. Only used with
      `objective="minimum"`.
    """
    objective: Literal["total", "minimum"] = "total"
    tiebreak: EgalitarianTiebreak = "leximin"

    @property
    def display_name(self) -> str:
        if self.objective == "minimum":
            return "E-Monroe-Exact" + _TIEBREAK_DISPLAY_SUFFIX[self.tiebreak]
        return "U-Monroe-Exact"

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
            - `utility`: The utility of the voter for the candidate
        """
        num_agents = rated_votes.shape[0]

        min_coalition_size = num_agents // slate_size
        max_coalition_size = -(-num_agents // slate_size)  # ceil(n / k)

        if self.objective == "minimum":
            slate, candidate_assignments = _exact_egalitarian_solution(
                rated_votes, slate_size, (min_coalition_size, max_coalition_size), self.tiebreak
            )
        else:
            slate, candidate_assignments = self._solve_total(rated_votes, slate_size, min_coalition_size, max_coalition_size)

        assignments: pd.DataFrame = pd.DataFrame(index=rated_votes.index)
        assignments["candidate_id"] = pd.Series(candidate_assignments, index=assignments.index, dtype=str)
        assignments["utility"] = [rated_votes.at[v, c] for v, c in zip(rated_votes.index, candidate_assignments)]

        return slate, assignments

    @staticmethod
    def _solve_total(
        rated_votes: pd.DataFrame,
        slate_size: int,
        min_coalition_size: int,
        max_coalition_size: int,
    ) -> tuple[list[str], list[str]]:
        """Solve the utilitarian integer program and return the selected slate and each agent's assigned statement."""
        num_agents = rated_votes.shape[0]
        num_statements = rated_votes.shape[1]
        U = rated_votes.to_numpy()  # Utility matrix

        prob = pulp.LpProblem("Maximize_total_Monroe_Utility", pulp.LpMaximize)

        # Decision variables
        x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(num_statements)]  # Whether to select statement j
        y = [
            [pulp.LpVariable(f"y_{i}_{j}", cat="Binary") for j in range(num_statements)]  # Agent i assigned to statement j
            for i in range(num_agents)
        ]
        agent_utility = [pulp.lpSum(U[i][j] * y[i][j] for j in range(num_statements)) for i in range(num_agents)]
        prob += pulp.lpSum(agent_utility), "TotalUtility"

        # Select exactly k statements
        prob += pulp.lpSum(x[j] for j in range(num_statements)) == slate_size, "Select_k_Statements"

        # Each agent is assigned to exactly one statement
        for i in range(num_agents):
            prob += pulp.lpSum(y[i][j] for j in range(num_statements)) == 1, f"Assign_Agent_{i}"

        # An agent can only be assigned to a selected statement
        for i in range(num_agents):
            for j in range(num_statements):
                prob += y[i][j] <= x[j], f"Assign_Agent_{i}_to_Selected_Statement_{j}"

        # Each selected statement gets floor(n/k) or ceil(n/k) agents, and unselected statements get none.
        # Combined with each agent being assigned once, exactly (n mod k) statements get the larger size.
        for j in range(num_statements):
            coalition_size = pulp.lpSum(y[i][j] for i in range(num_agents))
            prob += coalition_size >= min_coalition_size * x[j], f"Min_Coalition_Size_{j}"
            prob += coalition_size <= max_coalition_size * x[j], f"Max_Coalition_Size_{j}"

        prob.solve(pulp.PULP_CBC_CMD(msg=0, gapRel=1e-9, gapAbs=1e-12))
        if pulp.LpStatus[prob.status] != "Optimal":
            raise RuntimeError(f"{prob.name} not solved to optimality: {pulp.LpStatus[prob.status]}")

        slate = [rated_votes.columns[j] for j in range(num_statements) if pulp.value(x[j]) > 0.5]
        candidate_assignments: list[str] = []
        for i in range(num_agents):
            for j in range(num_statements):
                if pulp.value(y[i][j]) > 0.5:  # Assignment made
                    candidate_assignments.append(rated_votes.columns[j])
        return slate, candidate_assignments


@dataclass(frozen=True)
class ReweightedRangeVoting(VotingAlgorithm):
    """
    Reweighted Range Voting (RRV) algorithm.
    
    RRV works by iteratively selecting candidates with the highest weighted scores,
    then reducing the ballot weights of voters who "got their way" by having a 
    highly-rated candidate selected.
    
    # Arguments
    - `k: float = 1.0`: Positive constant used in the reweighting formula.
      Higher values of k make the reweighting less aggressive.
    - `max_rating: Optional[float] = None`: Maximum allowed rating value.
      If None, uses the maximum value found in the rated_votes data.
      If specified, uses this value in the reweighting formula.
    """
    k: float = 1.0
    max_rating: Optional[float] = None

    
    @property
    def display_name(self) -> str:
        return f"RRV(k={self.k})"

    @override
    def vote(
        self,
        rated_votes: pd.DataFrame,
        slate_size: int,
    ) -> tuple[list[str], pd.DataFrame]:
        """
        Select a slate of candidates using Reweighted Range Voting.
        
        # Arguments
        - `rated_votes: pd.DataFrame`: Utility of each voter (rows) for each candidate (columns)
        - `slate_size: int`: The number of candidates to be selected
        
        # Returns
        - `slate: list[str]`: The slate of candidates to be selected
        - `assignments: pd.DataFrame`: The assignments of the candidates to the voters with the following columns:
            - `candidate_id`: The candidate to which the voter is assigned
            - `utility`: The utility of the voter for the assigned candidate
            - `weight`: The final weight of each voter's ballot
        """
        if slate_size <= 0:
            return [], pd.DataFrame(index=rated_votes.index, columns=["candidate_id", "utility", "weight"])
        
        if slate_size > len(rated_votes.columns):
            slate_size = len(rated_votes.columns)
        
        # Initialize ballot weights to 1.0 for all voters
        weights = pd.Series(1.0, index=rated_votes.index)
        slate: list[str] = []
        
        # Get the maximum allowed score
        if self.max_rating is not None:
            max_score = self.max_rating
        else:
            # Use the maximum value found in the data (assuming all scores are non-negative)
            max_score = rated_votes.max().max()
            if max_score <= 0:
                max_score = 1.0  # Avoid division by zero
        
        # Iteratively select candidates
        remaining_candidates = set(rated_votes.columns)
        
        for _ in range(slate_size):
            if not remaining_candidates:
                break
                
            # Calculate weighted scores for each remaining candidate
            best_candidate = None
            best_score = float('-inf')
            
            for candidate in remaining_candidates:
                # Sum of weighted scores for this candidate
                weighted_score = (weights * rated_votes[candidate]).sum()
                
                if weighted_score > best_score:
                    best_score = weighted_score
                    best_candidate = candidate
            
            if best_candidate is None:
                break
                
            # Add the best candidate to the slate
            slate.append(best_candidate)
            remaining_candidates.remove(best_candidate)
            
            # Update ballot weights using the RRV formula
            # weight = k / (k + SUM/MAX)
            # where SUM is the sum of scores given to winners-so-far
            sum_scores_for_winners = rated_votes[slate].sum(axis=1)
            weights = self.k / (self.k + sum_scores_for_winners / max_score)
        
        # Create assignments by assigning each voter to their highest-utility candidate in the slate
        if slate:
            assignments = pd.DataFrame(index=rated_votes.index)
            assignments["candidate_id"] = rated_votes[slate].idxmax(axis=1)
            assignments["utility"] = rated_votes[slate].max(axis=1)
            assignments["weight"] = weights
        else:
            # Handle edge case where no candidates were selected
            assignments = pd.DataFrame(index=rated_votes.index)
            assignments["candidate_id"] = pd.Series([None] * len(rated_votes.index), 
                                                  index=rated_votes.index, dtype=str)
            assignments["utility"] = pd.Series(0.0, index=rated_votes.index)
            assignments["weight"] = weights
        
        return slate, assignments