import json
import os
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

import pandas as pd

from generative_social_choice.utils.helper_functions import get_base_dir_path
from generative_social_choice.queries.query_interface import LLMLog
from generative_social_choice.queries.query_chatbot_personalization import ChatbotPersonalizationAgent


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
        ratings_file: Optional[Path] = None,
        log_file: Optional[Path] = None,
    ) -> tuple[List[Rating], list[LLMLog]]:
    # If we already have some completions, consider them
    if ratings_file is not None and ratings_file.exists():
        ratings = [Rating(**json.loads(line)) for line in open(ratings_file, "r")]
    else:
        ratings = []
    
    logs = []
    for statement in statements:
        new_ratings, log = generate_ratings(agents=agents, statement=statement, ratings=ratings, verbose=verbose)

        ratings.extend(new_ratings)
        logs.extend(log)

        if ratings_file is not None:
            # Write new ratings to disk
            with open(ratings_file, "a") as f:
                for rating in new_ratings:
                    f.write(json.dumps(rating.to_dict())+'\n')
        if log_file is not None:
            # Write log to file
            log_df = pd.DataFrame.from_records(logs)
            if os.path.exists(log_file) and len(''.join([line for line in open(log_file, "r")]).strip()):
                log_df = pd.concat([pd.read_csv(log_file), log_df])
            log_df.to_csv(log_file, index=False)

    return ratings, logs