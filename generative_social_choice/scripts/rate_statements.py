import random
import json
import os
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

import pandas as pd

from generative_social_choice.utils.helper_functions import get_base_dir_path
from generative_social_choice.queries.query_interface import Agent, LLMLog
from generative_social_choice.queries.query_chatbot_personalization import ChatbotPersonalizationAgent, SimplePersonalizationAgent


RATINGS_FILE = get_base_dir_path() / "data/demo_data/TEST_ratings.jsonl"
STATEMENTS_FILE = get_base_dir_path() / "data/demo_data/2025-01-22-193423_statement_generation/statement_generation_raw_output.csv"
LOG_FILE = get_base_dir_path() / "data/demo_data/TEST_ratings_logs.csv"
UTILITY_MATRIX_FILE = get_base_dir_path() / "data/demo_data/TEST_utility_matrix.csv"
STATEMENT_ID_FILE = get_base_dir_path() / "data/demo_data/TEST_utility_matrix_statements.csv"

@dataclass
class Rating:
    """
    Approval rating of a single agent for a single statement

    # Arguments
    - `agent_id: str`: ID of the agent as given in the survey data
    - `statement: str`: Statement that is rated
    - `approval: float`: Numeric approval score
    """
    agent_id: str
    statement: str
    approval: float

    def to_dict(self):
        return {"agent_id": self.agent_id, "statement": self.statement, "approval": self.approval}


def get_agents(model: Optional[str] = None) -> List[ChatbotPersonalizationAgent]:
    """Utility function to get all agents based on survey data and summaries"""
    if model is None:
        model = "gpt-4o-mini"
    
    df = pd.read_csv(get_base_dir_path() / "data/chatbot_personalization_data.csv")
    df = df[df["sample_type"] == "generation"]
    agent_id_to_summary = (
        pd.read_csv(get_base_dir_path() / "data/user_summaries_generation.csv")
        .set_index("user_id")["summary"]
        .to_dict()
    )

    agents = []
    for id in df.user_id.unique():
        agent = ChatbotPersonalizationAgent(
            id=id,
            survey_responses=df[df.user_id == id],
            summary=agent_id_to_summary[id],
            model=model,
        )
        agents.append(agent)
    return agents

def rate_statement(agent: ChatbotPersonalizationAgent, statement: str) -> tuple[float, list[LLMLog]]:
    """Predict the rating the given agent would assign to the statement"""
    # NOTE This is just the agent method, so might drop this function
    return agent.get_approval(statement=statement)

def generate_ratings(
        agents: List[ChatbotPersonalizationAgent],
        statement: str,
        ratings: Optional[List[Rating]] = None,
        verbose: bool = False,
    ) -> tuple[List[Rating], list[LLMLog]]:
    if verbose:
        print(f"Generating approval ratings for the statement '{statement}' ...")

    # First check for which agents we already have approval ratings
    covered_agents = set()
    if ratings is not None:
        for rating in ratings:
            if rating.statement.strip()==statement.strip():
                covered_agents.add(rating.agent_id)
    # Check if the statement is part of the original 6 statements
    for agent in agents:
        if statement in agent.survey_responses["statement"].to_list():
            covered_agents.add(agent.id)

    logs = []
    new_ratings = []
    for agent in agents:
        # Skip agents who already have approval ratings
        if agent.id in covered_agents:
            continue

        if verbose:
            print(f"- Computing approval ratings for user '{agent.id}' ...")

        # Otherwise compute the approval rating
        approval, log = agent.get_approval(statement=statement)
        rating = Rating(agent_id=agent.id, statement=statement, approval=approval)
        new_ratings.append(rating)
        logs.extend(log)
        covered_agents.add(agent.id)

    return new_ratings, logs


def complete_ratings(
        agents: List[ChatbotPersonalizationAgent],
        statements: List[str],
        verbose: bool = False,
    ) -> tuple[List[Rating], list[LLMLog]]:
    # If we already have some completions, consider them
    if RATINGS_FILE.exists():
        ratings = [Rating(**json.loads(line)) for line in open(RATINGS_FILE, "r")]
    else:
        ratings = []
    
    logs = []
    for statement in statements:
        new_ratings, log = generate_ratings(agents=agents, statement=statement, ratings=ratings, verbose=verbose)

        ratings.extend(new_ratings)
        logs.extend(log)

        # Write new ratings to disk
        with open(RATINGS_FILE, "a") as f:
            for rating in new_ratings:
                f.write(json.dumps(rating.to_dict())+'\n')
        # Write log to file
        log_df = pd.DataFrame.from_records(logs)
        if os.path.exists(LOG_FILE) and len(''.join([line for line in open(LOG_FILE, "r")]).strip()):
            log_df = pd.concat([pd.read_csv(LOG_FILE), log_df])
        log_df.to_csv(LOG_FILE, index=False)

    return ratings, logs

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
        utility_matrix_file: Optional[Path] = None,
        statement_id_file: Optional[Path] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create the utility matrix for the given agents and statements
    
    # Arguments
    - `agents: List[ChatbotPersonalizationAgent]`: Agents to consider in the utility matrix
    - `statements: List[str]`: Statements to include in the utility matrix
    - `prepend_survey_statements: bool = True`: If True, the initial 6 statements from the survey
      are included at the beginning
    - `utility_matrix_file: Optional[Path] = None`: If a path is given, write the dataframe with the utility matrix
      to this path (in CSV format).
    - `statement_id_file: Optional[Path] = None`: If a path is given, write the dataframe with the statement IDs
      to this path (in CSV format).

    # Returns
    - `utility_matrix: pd.DataFrame`: The utility matrix, with agent IDs as index and statement IDs as column names
    - `statement_ids: pd.DataFrame`: Dataframe with statement IDs as index and statements in column "statement"
    """
    # NOTE That this only predicts missing entries
    ratings, logs = complete_ratings(agents=agents, statements=statements, verbose=True)

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

if __name__=="__main__":
    agents = get_agents(model="gpt-4o-mini")

    # Subsampling for testing purposes
    #agents = random.sample(agents, 5)
    agents = agents[:4]

    # Read generated statements
    statements = pd.read_csv(STATEMENTS_FILE)["statement"].to_list()

    # Subsampling for testing purposes
    #statements = random.sample(statements, 2)
    statements = statements[:2]

    create_utility_matrix(
        agents=agents,
        statements=statements,
        utility_matrix_file=UTILITY_MATRIX_FILE,
        statement_id_file=STATEMENT_ID_FILE,
    )

#TODO Move functions to other files
#TODO Add basic testing