from typing import Optional, List
from pathlib import Path

import pandas as pd

from generative_social_choice.queries.query_chatbot_personalization import ChatbotPersonalizationAgent, SimplePersonalizationAgent
from generative_social_choice.ratings.rating_generation import complete_ratings, Rating


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