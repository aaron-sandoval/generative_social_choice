import os
import random
import pandas as pd
import numpy as np

from typing import Optional

from generative_social_choice.utils.helper_functions import (
    get_base_dir_path,
    get_time_string,
)
from generative_social_choice.queries.query_chatbot_personalization import generate_fewshot_prompt_template, ChatbotPersonalizationAgent

SURVEY_DATA_PATH = get_base_dir_path() / "data/chatbot_personalization_data.csv"
SUMMARY_DATA_PATH = get_base_dir_path() / "data/user_summaries_generation.csv"


def check_survey_data():
    df = pd.read_csv(SURVEY_DATA_PATH)
    #print(len(df))
    #print(df.iloc[10])

    # Format
    # - question_type "multiple choice + text" (detailed_question_type "rating_statement") has
    #   - numeric ratings in column choice_numeric of the statement in column statement. (Should be same 6 statements for everyone.)
    #   - In the column text, there is an additional explanation from the user.
    # - detailed_question_type "general opinion" has additional opinions in field text
    # - detailed_question_type "example scenario" describes a scenario in question_text, and has thoughts on it in field text
    # - Rows with question_type "reading" can be skipped

    # Checking how their discriminative prompt is created
    user_data = df[df["user_id"]=="generation1"]
    #result = generate_fewshot_prompt_template(survey_responses=user_data, approval_levels=ChatbotPersonalizationAgent.approval_levels)
    #print("\nRESULT")
    #print(result)
    # But how to call this on a new statement?
    # -> Use class ChatbotPersonalizationAgent. Init giving relevant responses and summary, then call the respective method with statement as arg

    # User embeddings can now be done based on
    # - Their statements ratings (simplest but ignoring free-form texts)
    # - Embedding everything with an LLM
    # - Embedding the summary of their responses with an LLM or other NLP methods

    # Just create a simple vector from statement ratings for test purposes:
    statements = []  # To fix ordering
    user_ratings = {}
    for row in user_data.to_records():
        if row["statement"]!=row["statement"]:
            continue  # NaN
        user_ratings[row["statement"]] = row["choice_numeric"]

        if len(statements)<6:
            statements.append(row["statement"])
        else:
            assert row["statement"] in statements
    user_ratings = np.array([user_ratings[statement] for statement in statements])
    print(user_ratings)

def dummy():
    survey_df = pd.read_csv(SURVEY_DATA_PATH)
    df = pd.read_csv(SUMMARY_DATA_PATH)

    summaries = df.set_index("user_id")["summary"].to_dict()  # {user_id: summary}
    
    # So how to generate new statements? (Debug that stuff with a small subset of responses and mini version)
    # Toy method: Return random strings
    # Baseline method: Just put all into into an LLM call
    # Other baseline: Randomly sample some subset and call the baseline method for that subset
    # Based on clustering:
    # 1. Compute embeddings (e.g. based on ratings, summaries or whole response) - check what they did in v3
    #    - Fish et al. v3 did this by randomly selecting 50 statements from the generated summaries and ask the LLM
    #      to rate on 7 point scale how much an agent is aligned, giving a 50d vector for each agent
    # 2. Cluster based on number of statements to generate
    # 3. For each cluster, generate one or several statements (can use the baseline method or similar one)
    from generative_social_choice.statements.statement_generation import DummyStatementGeneration

    generator = DummyStatementGeneration()
    statements = generator.generate(survey_responses=survey_df, summaries=summaries, num_statements=5)
    print(statements)

    # Output into data/demo_data with timestring prepended
    timestring = get_time_string()
    dirname = (
        get_base_dir_path()
        / "data/demo_data"
        / f"{timestring}_statement_generation"
    )
    #os.makedirs(dirname)

    #TODO Add extra info like generation method name and IDs of people that were considered?
    #df.to_csv(dirname / "statement_generation_raw_output.csv", mode="append")

from generative_social_choice.queries.query_chatbot_personalization import ChatbotPersonalizationGenerator

def generate_statements(num_agents: Optional[int] = None, model: str = "default"):
    #NOTE: This uses stuff from paper_replication.generate_slate
    gen_query_model_arg = {"model": model} if model != "default" else {}

    # Set up agents

    df = pd.read_csv(get_base_dir_path() / "data/chatbot_personalization_data.csv")
    df = df[df["sample_type"] == "generation"]
    agent_id_to_summary = (
        pd.read_csv(get_base_dir_path() / "data/user_summaries_generation.csv")
        .set_index("user_id")["summary"]
        .to_dict()
    )

    agents = []
    for id in df.user_id.unique():
        agent = SimplePersonalizationAgent(
            id=id,
            survey_responses=df[df.user_id == id],
            summary=agent_id_to_summary[id],
        )
        agents.append(agent)

    # Subsample agents
    if num_agents is not None:
        agents = random.sample(agents, num_agents)

    # Set up generators

    random.seed(0)
    # Due to a implementation oversight, NN generators didn't set their own
    # local random seed. So, their behavior was determined by this global seed

    generators = [
        DummyGenerator(),
        #ChatbotPersonalizationGenerator(
        #    seed=0, gpt_temperature=0, **gen_query_model_arg
        #),
        #ChatbotPersonalizationGenerator(
        #    seed=0, gpt_temperature=1, **gen_query_model_arg
        #),
    ]
    # ---- End of copied code -----

    # Now for all the generators, generate statements, then write the results to some file
    results = []
    for generator in generators:
        statements, logs = generator.generate(agents=agents)
        #TODO Better to store arguments passed to init of the generator as well? And perhaps some info on which agents were used
        results.extend([{"statement": statement, "generator": generator.__class__.__name__} for statement in statements])
    
    # Output into data/demo_data with timestring prepended
    print("Writing results to file ...")
    timestring = get_time_string()
    dirname = (
        get_base_dir_path()
        / "data/demo_data"
        / f"{timestring}_statement_generation"
    )
    os.makedirs(dirname)

    pd.DataFrame.from_records(results).to_csv(dirname / "statement_generation_raw_output.csv", mode="a", index=False)
    #TODO Store the logs somewhere as well!
    print("Done.")

import random
import string
from typing import Tuple, List
from generative_social_choice.queries.query_interface import Generator, Agent
from generative_social_choice.utils.gpt_wrapper import LLMLog

class DummyGenerator(Generator):
    """Dummy method that returns random strings as new statements.
    
    Use for test purposes only!"""

    def __init__(self, num_statements: int=5, statement_length: int=20):
        self.num_statements = num_statements
        self.statement_length = statement_length

    def generate(self, agents: List[Agent]) -> Tuple[List[str], List[LLMLog]]:
        """
        Returns random strings of fixed length with letters and whitespace.
        """
        statements = []
        for _ in range(self.num_statements):
            new_statement = ''.join(random.choices(string.ascii_letters + " ", k=self.statement_length))
            statements.append(new_statement)
        return statements, []

from generative_social_choice.queries.query_interface import Agent
from generative_social_choice.utils.gpt_wrapper import LLMLog

class SimplePersonalizationAgent(Agent):
    """Simple agent representation which doesn't require connecting to any LLM
    but can't be used to get approvals."""

    def __init__(
        self,
        *,
        id: str,
        survey_responses: pd.DataFrame,
        summary: str,
    ):
        self.id = id
        self.survey_responses = survey_responses
        self.summary = summary

    def get_id(self):
        return self.id

    def get_description(self):
        return self.summary

    def get_approval(
        self, statement: str, use_logprobs: bool = True
    ) -> tuple[float, list[LLMLog]]:
        raise NotImplementedError()

if __name__=="__main__":
    # Should we use their Generator interface to generate statements? -> Check the output format and how it's used in the pipeline
    # In the pipeline, it initializes a bunch of generators and then does the heavy lifting in slate_generation generate_slate_ensemble_greedy
    # In there, several rounds are done where all generators are called on agents not yet assigned
    # -> Make sure this works with caching and doesn't do unnecessary LLM calls to DISC, but otherwise seems good to use!
    #    -> How about implementing another Agent subclass with a dummy approval method that raises an exception if called?
    #       Do use the Generator class then, so we can probably just take their implementation as the baseline class!
    # NOTE For embeddings we might want to use caching still, like writing to some file (but also, this is rather optional!)
    generate_statements(num_agents=5, model="gpt-4o-mini")