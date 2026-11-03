from dataclasses import dataclass
from typing import Literal, Optional, List
from pathlib import Path
import json
import re

import pandas as pd

from generative_social_choice.queries.simple_personalization_agent import SimplePersonalizationAgent
from generative_social_choice.queries.query_chatbot_personalization import ChatbotPersonalizationAgent
from generative_social_choice.ratings.rating_generation import complete_ratings, Rating
from generative_social_choice.utils.helper_functions import get_base_dir_path


def get_initial_statements(agents: List[ChatbotPersonalizationAgent | SimplePersonalizationAgent]) -> List[str]:
    """Get the initial statements for which approval ratings were collected in the study
    
    Note that this could be done more efficiently based on the dataframe, but since we usually have
    agents initialized anyway, we are using this path."""
    statements = set()

    for agent in agents:
        for statement in agent.survey_responses["statement"].to_list():
            if statement==statement and statement not in statements:
                statements.add(statement)
    return sorted(list(statements))


def create_utility_matrix(
        agents: List[ChatbotPersonalizationAgent],
        statements: List[str],
        prepend_survey_statements: bool = True,
        ratings_file: Optional[Path] = None,
        log_file: Optional[Path] = None,
        utility_matrix_file: Optional[Path] = None,
        statement_id_file: Optional[Path] = None,
        verbose: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create the utility matrix for the given agents and statements
    
    # Arguments
    - `agents: List[ChatbotPersonalizationAgent]`: Agents to consider in the utility matrix
    - `statements: List[str]`: Statements to include in the utility matrix
    - `prepend_survey_statements: bool = True`: If True, the initial 6 statements from the survey
      are included at the beginning
    - `ratings_file: Optional[Path] = None`: If a path is given, read available ratings from that file if it exists,
      and write any newly computed ratings into the file (creating a new file if there is none).
      If no path is given, all ratings are generated and they are not stored to disk.
    - `log_file: Optional[Path] = None`: If a path is given, write any logs created during rating generation
      into this file (using append mode).
    - `utility_matrix_file: Optional[Path] = None`: If a path is given, write the dataframe with the utility matrix
      to this path (in CSV format).
    - `statement_id_file: Optional[Path] = None`: If a path is given, write the dataframe with the statement IDs
      to this path (in CSV format).
    - `verbose: bool = False`: If True, print some additional information

    # Returns
    - `utility_matrix: pd.DataFrame`: The utility matrix, with agent IDs as index and statement IDs as column names
    - `statement_ids: pd.DataFrame`: Dataframe with statement IDs as index and statements in column "statement"
    """
    # Normalize statements
    statements = list(set([statement.strip() for statement in statements]))

    # NOTE This only predicts missing entries if ratings_file is given
    ratings, logs = complete_ratings(
        agents=agents,
        statements=statements,
        verbose=verbose,
        ratings_file=ratings_file,
        log_file=log_file,
    )

    # Verify with initial statement that it's not recomputed (make a test case for this perhaps)
    #statement = "The most important rule for chatbot personalization is complete avoidance; it's a ticking time bomb for privacy invasion. For example, a chatbot revealing someone's sexual orientation could be life-threatening in certain countries."
    #complete_ratings(agents=agents, statements=[statement], verbose=True)

    # Add initial statements
    # Get corresponding approvals as Ratings
    if prepend_survey_statements:
        for statement in get_initial_statements(agents=agents)[::-1]:
            for agent in agents:
                approval = agent.survey_responses[agent.survey_responses["statement"]==statement]["choice_numeric"].to_list()[0]
                ratings.append(Rating(agent_id=agent.id, statement=statement, approval=approval))

            # Add initial statements to the beginning of the list
            statements = [statement] + statements

    statement_ids = [f"s{i+1}" for i in range(len(statements))]
    id_lookup = {statement: statement_id for statement, statement_id in zip(statements, statement_ids)}

    # Organize ratings as lookup dict
    lookup_ratings = {}
    for rating in ratings:
        if rating.statement not in statements:
            continue
        if rating.agent_id not in lookup_ratings:
            lookup_ratings[rating.agent_id] = {}
        lookup_ratings[rating.agent_id][id_lookup[rating.statement]] = rating.approval

    # Now create a dataframe with the utility matrix
    utilities = {}
    for agent in agents:
        utilities[agent.id] = lookup_ratings[agent.id]
    # We transpose to get statements in columns, and index to get same ordering as in statement_ids
    utilities_df = pd.DataFrame(utilities).T[statement_ids]

    # Now write the utility matrix to a file, and also the statements,
    # so that we can convert back from ID to statement after assigning agents
    if utility_matrix_file is not None:
        utilities_df.to_csv(utility_matrix_file)
    statement_df = pd.DataFrame([{"statement": statement, "id": statement_id} for statement, statement_id in zip(statements, statement_ids)]).set_index("id")
    if statement_id_file is not None:
        statement_df.to_csv(statement_id_file)

    return utilities_df, statement_df


def extract_voter_utilities_from_info_csv(info_csv_path: Path, assigned_utilities_only: bool = True) -> pd.DataFrame:
    """
    Extracts a DataFrame of utilities from the info.csv generated by generate_slate.py.
    If assigned_utilities_only is False, all available utilities are included to form a (potentially incomplete) utility matrix.

    Each row represents a voter, with columns for
    the matched candidate index and the approval value for that candidate.
    In info.csv, the columns are of the form 'approval_generationX' and 'matched_generationX', each row containing a candidate.
    Example: generative_social_choice\data\demo_data\2025-05-02-205127__generate_slate_via_openai_embeddings\info.csv

    Args:
        info_csv_path (Path): The path to the info.csv file generated by generate_slate.py.
        assigned_utilities_only (bool): If True, only the utilities for the assigned statements are included.

    Returns:
        pd.DataFrame: DataFrame with columns ['Voter', 'Candidate', 'Utility'].
    """
    df = pd.read_csv(info_csv_path)
    voter_ids = [
        re.match(r"matched_generation(\d+)", col).group(1)
        for col in df.columns if col.startswith("matched_generation")
    ]

    records = []
    for voter_id in voter_ids:
        matched_col = f"matched_generation{voter_id}"
        approval_col = f"approval_generation{voter_id}"

        # Find the candidate (row) where this voter is matched (non-null)
        matched_idx = df[matched_col].first_valid_index()
        if matched_idx is not None:
            approval_value = df.loc[matched_idx, approval_col]
            records.append({
                "Voter": f"generation{voter_id}",
                "candidate_id": f"s{matched_idx}",
                "utility": approval_value
            })

    return pd.DataFrame(records).set_index("Voter").sort_index()



def get_baseline_run_dirs(run_ids: list[int] | None = None, embedding_type: Literal["llm", "seed_statement"] = "llm") -> dict[int, Path]:
    """
    Get the output directories of the baseline generate_slate.py runs.

    Runs are identified by their zero-indexed position among the sorted run directories,
    so no other directories may be added next to them.

    Args:
        run_ids: List of zero-indexed directory indices to load. If None, loads all directories.
        embedding_type: "llm" for generate_slate_results_baseline, "seed_statement" for generate_slate_results_openai_embeddings

    Returns:
        Dict mapping run_id to the run directory.
    """
    # Choose directory based on embedding type
    if embedding_type == "llm":
        baseline_dir = get_base_dir_path() / "data/demo_data/generate_slate_results_baseline"
    elif embedding_type == "seed_statement":
        baseline_dir = get_base_dir_path() / "data/demo_data/generate_slate_results_openai_embeddings"
    else:
        raise ValueError(f"Invalid embedding_type: {embedding_type}. Must be 'llm' or 'seed_statement'")

    # Get all directories sorted by name
    all_dirs = sorted([d for d in baseline_dir.iterdir() if d.is_dir()])

    # Select directories based on run_ids
    if run_ids is None:
        run_ids = range(len(all_dirs))
    return {i: all_dirs[i] for i in run_ids}


def get_baseline_generate_slate_results(run_ids: list[int] | None = None, embedding_type: Literal["llm", "seed_statement"] = "llm") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load baseline generate slate results from data/demo_data directories.
    
    Args:
        run_ids: List of zero-indexed directory indices to load. If None, loads all directories.
        embedding_type: "llm" for generate_slate_results_baseline, "seed_statement" for generate_slate_results_openai_embeddings
        
    Returns:
        Tuple of (utilities_df, assignments_df) where column names are run_ids.
    """
    run_dirs = get_baseline_run_dirs(run_ids, embedding_type)
    selected_dirs = list(run_dirs.values())
    column_names = list(run_dirs.keys())

    utilities_data = []
    assignments_data = []
    
    for i, dir_path in enumerate(selected_dirs):
        info_csv_path = dir_path / "info.csv"
        
        if not info_csv_path.exists():
            raise FileNotFoundError(f"info.csv not found in {dir_path}")
        
        # Extract utilities using the helper function
        utilities_df = extract_voter_utilities_from_info_csv(info_csv_path, assigned_utilities_only=False)
        utilities_data.append(utilities_df['utility'])
        
        # Create assignments series (candidate_id for each voter)
        assignments_data.append(utilities_df['candidate_id'])
    
    # Combine into DataFrames with run_ids as column names
    utilities = pd.DataFrame({
        col_name: data for col_name, data in zip(column_names, utilities_data)
    })
    
    assignments = pd.DataFrame({
        col_name: data for col_name, data in zip(column_names, assignments_data)
    })
    
    return utilities, assignments

# The baseline runs only rated each round's candidates by the voters unassigned in that round.
# The missing ratings are stored next to info.csv in these files, leaving the run outputs untouched.
COMPLETION_RATINGS_FILENAME = "completion_ratings.jsonl"
COMPLETION_LOGS_FILENAME = "completion_logs.csv"


def read_baseline_info_csv(run_dir: Path) -> pd.DataFrame:
    """
    Read the info.csv of a baseline run, indexed by the candidate statements as they were rated.

    info.csv was written with every newline in a statement turned into "\r\n", so this is undone
    to recover the statements which were passed to the discriminative queries.
    """
    info = pd.read_csv(run_dir / "info.csv", index_col=0)
    info.index = [statement.replace("\r\n", "\n") for statement in info.index]
    return info


@dataclass
class BaselineProfile:
    """
    Complete ballot profile of a baseline run, together with the outcome of the baseline pipeline.

    Candidate IDs are `s{row index in info.csv}`, as in `extract_voter_utilities_from_info_csv`.

    # Arguments
    - `utilities: pd.DataFrame`: Utility of each voter (rows) for each candidate (columns)
    - `statements: pd.Series`: Statement of each candidate ID
    - `round_added: pd.Series`: Round of the pipeline in which each candidate was generated
    - `slate: list[str]`: Candidate IDs selected by the pipeline, in the order of selection
    - `assignments: pd.DataFrame`: The pipeline's assignment, with columns `candidate_id` and `utility`
    """
    utilities: pd.DataFrame
    statements: pd.Series
    round_added: pd.Series
    slate: list[str]
    assignments: pd.DataFrame


def get_completed_baseline_utility_matrices(
        run_ids: list[int] | None = None,
        embedding_type: Literal["llm", "seed_statement"] = "llm",
        allow_incomplete: bool = False,
    ) -> dict[int, BaselineProfile]:
    """
    Load the complete ballot profiles of baseline runs by combining the ratings in info.csv
    with the missing ratings collected by `scripts/complete_baseline_ratings.py`.

    Args:
        run_ids: List of zero-indexed directory indices to load. If None, loads all directories.
        embedding_type: See `get_baseline_run_dirs`
        allow_incomplete: If False, raise an error if any rating is still missing.

    Returns:
        Dict mapping run_id to the BaselineProfile of that run.
    """
    profiles = {}
    for run_id, run_dir in get_baseline_run_dirs(run_ids, embedding_type).items():
        info = read_baseline_info_csv(run_dir)
        candidate_ids = [f"s{i}" for i in range(len(info))]
        statement_to_id = dict(zip(info.index, candidate_ids))

        approval_cols = [col for col in info.columns if col.startswith("approval_")]
        utilities = info[approval_cols].T
        utilities.index = [col.removeprefix("approval_") for col in approval_cols]
        utilities.columns = candidate_ids
        utilities = utilities.sort_index()

        ratings_file = run_dir / COMPLETION_RATINGS_FILENAME
        if ratings_file.exists():
            with open(ratings_file, "r", encoding="utf-8") as f:
                ratings = [Rating(**json.loads(line)) for line in f]
        else:
            ratings = []
        for rating in ratings:
            candidate_id = statement_to_id[rating.statement]
            if not pd.isna(utilities.loc[rating.agent_id, candidate_id]):
                raise ValueError(f"Rating of {rating.agent_id} for {candidate_id} in {run_dir} is also in info.csv")
            utilities.loc[rating.agent_id, candidate_id] = rating.approval

        num_missing = int(utilities.isna().sum().sum())
        if num_missing and not allow_incomplete:
            raise ValueError(f"{num_missing} ratings are missing in {run_dir}. Run scripts/complete_baseline_ratings.py first.")

        slate = [statement_to_id[s] for s in info["round_chosen"].dropna().sort_values().index]
        profiles[run_id] = BaselineProfile(
            utilities=utilities,
            statements=pd.Series(info.index, index=candidate_ids),
            round_added=pd.Series(info["round_added"].to_numpy(), index=candidate_ids),
            slate=slate,
            assignments=extract_voter_utilities_from_info_csv(run_dir / "info.csv"),
        )
    return profiles
