import random
import json
import os
from dataclasses import dataclass
from typing import Optional, List

import pandas as pd

from generative_social_choice.utils.helper_functions import get_base_dir_path
from generative_social_choice.queries.query_interface import Agent, LLMLog
from generative_social_choice.queries.query_chatbot_personalization import ChatbotPersonalizationAgent


RATINGS_FILE = get_base_dir_path() / "data/demo_data/TEST_ratings.jsonl"
STATEMENTS_FILE = get_base_dir_path() / "data/demo_data/2025-01-22-193423_statement_generation/statement_generation_raw_output.csv"
LOG_FILE = get_base_dir_path() / "data/demo_data/TEST_ratings_logs.csv"

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


def complete_ratings(agents: List[ChatbotPersonalizationAgent], statements: List[str], verbose: bool = False):
    # If we already have some completions, consider them
    if RATINGS_FILE.exists():
        existing_ratings = [Rating(**json.loads(line)) for line in open(RATINGS_FILE, "r")]
    else:
        existing_ratings = []
    
    logs = []
    for statement in statements:
        new_ratings, log = generate_ratings(agents=agents, statement=statement, ratings=existing_ratings, verbose=verbose)

        existing_ratings.extend(new_ratings)
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

    # NOTE That this only predicts missing entries
    complete_ratings(agents=agents, statements=statements, verbose=True)

    # Verify with initial statement that it's not recomputed (make a test case for this perhaps)
    #statement = "The most important rule for chatbot personalization is complete avoidance; it's a ticking time bomb for privacy invasion. For example, a chatbot revealing someone's sexual orientation could be life-threatening in certain countries."
    #complete_ratings(agents=agents, statements=[statement], verbose=True)

    #TODO! Now let's actually create the approval matrix!

#TODO Move functions to other files
#TODO Add basic testing