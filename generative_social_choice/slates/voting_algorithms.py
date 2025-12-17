from collections import defaultdict
from functools import cached_property
from itertools import combinations
import abc
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
    Utility transformation based on geometric series.
    
    This transformation takes a tunable parameter p>1 and maps utility u to f(u)
    such that
     (i) `f(u+1)-f(u)=p*(f(u+2)-f(u+1))` for all u>=0
     (ii) f(0) = 0
     (iii) f(1) = 1
     (iv) f(u) > 0 for all u>0
     
    The transformation is given by
    `f(u) = (1-(1/p)^u)/(1-(1/p))
    
    Higher values of p make total utility increasingly egalitarian.
    """
    p: float=1.5

    @property
    def display_name(self) -> str:
        return f"p={self.p}"

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
            after transformation
        """
        transform_fct = lambda u: (1-(1/self.p)**u)/(1-(1/self.p))
        transformed_rated_votes = rated_votes.apply(transform_fct)
        return transformed_rated_votes

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
        return f"Exact{f'({self.utility_transform.display_name})' if self.utility_transform is not None else ''}"
        
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


@dataclass(frozen=True)
class ExactTotalUtilityMaximizationBruteSearch(VotingAlgorithm):
    """Brute-force search over all slates of size k to maximize total utility."""

    utility_transform: Optional[UtilityTransformation] = None

    @property
    def display_name(self) -> str:
        return f"ExactBrute{f'({self.utility_transform.display_name})' if self.utility_transform is not None else ''}"

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
        return f"LP{f'({self.utility_transform.display_name})' if self.utility_transform is not None else ''}"

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
        return f"Greedy{f'({self.utility_transform.display_name})' if self.utility_transform is not None else ''}"

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
            new_total_utility = current_total_utility
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
            if new_total_utility<=current_total_utility:
                break

            # Update the slate and assignments
            slate.append(best_candidate)
            assignments["candidate_id"] = rated_votes.loc[:, slate].idxmax(axis=1)
            assignments["utility"] = rated_votes.loc[:, slate].max(axis=1)

        #TODO If a utility transformation was applied, assignments["utilities"] won't match the original utilities
        return slate, assignments


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