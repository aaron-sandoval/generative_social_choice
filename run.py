import os
import random
import pandas as pd
import numpy as np

from typing import Optional

from generative_social_choice.utils.helper_functions import (
    get_base_dir_path,
    get_time_string,
)

SURVEY_DATA_PATH = get_base_dir_path() / "data/chatbot_personalization_data.csv"
SUMMARY_DATA_PATH = get_base_dir_path() / "data/user_summaries_generation.csv"

# Notes on survey data format
# - question_type "multiple choice + text" (detailed_question_type "rating_statement") has
#   - numeric ratings in column choice_numeric of the statement in column statement. (Should be same 6 statements for everyone.)
#   - In the column text, there is an additional explanation from the user.
# - detailed_question_type "general opinion" has additional opinions in field text
# - detailed_question_type "example scenario" describes a scenario in question_text, and has thoughts on it in field text
# - Rows with question_type "reading" can be skipped

from generative_social_choice.queries.query_interface import Agent
from generative_social_choice.statements.statement_generation import SimplePersonalizationAgent

def get_simple_agents():
    """Utility function to get all agents based on survey data and summaries"""
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
    return agents


from generative_social_choice.statements.statement_generation import (
    DummyGenerator,
    NamedChatbotPersonalizationGenerator,
    LLMGenerator,
)

def generate_statements(num_agents: Optional[int] = None, model: str = "default"):
    gen_query_model_arg = {"model": model} if model != "default" else {}

    # Set up agents
    agents = get_simple_agents()

    # Subsample agents
    if num_agents is not None:
        agents = random.sample(agents, num_agents)

    # Set up generators

    random.seed(0)
    # Due to a implementation oversight, NN generators didn't set their own
    # local random seed. So, their behavior was determined by this global seed

    generators = [
        DummyGenerator(),
        #NamedChatbotPersonalizationGenerator(
        #    seed=0, gpt_temperature=0, **gen_query_model_arg
        #),
        #LLMGenerator(
        #    seed=0, gpt_temperature=0, **gen_query_model_arg
        #),
        #NamedChatbotPersonalizationGenerator(
        #    seed=0, gpt_temperature=1, **gen_query_model_arg
        #),
    ]

    # Now for all the generators, generate statements, then write the results to some file
    results = []
    logs = []
    for generator in generators:
        statements, lgs = generator.generate(agents=agents)
        results.extend([{"statement": statement, "generator": generator.name, "agents": sorted([agent.id for agent in agents])}
                        for statement in statements])
        logs.extend(lgs)
    
    # Output into data/demo_data with timestring prepended
    print("Writing results to file ...")
    timestring = get_time_string()
    dirname = (
        get_base_dir_path()
        / "data/demo_data"
        #/ f"{timestring}_statement_generation"
        / "TEST_statement_generation"  #TODO use other version after debugging
    )
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    result_file = dirname / "statement_generation_raw_output.csv"
    # NOTE: Pandas can write csvs in append mode, but that is error prone as previous header entries
    # aren't checked.
    result_df = pd.DataFrame.from_records(results)
    # Existing but empty files also create issues, so we check for that
    if os.path.exists(result_file) and len(''.join([line for line in open(result_file, "r")]).strip()):
        result_df = pd.concat([pd.read_csv(result_file), result_df])
    result_df.to_csv(result_file, index=False)

    log_file = dirname / "statement_generation_logs.csv"
    log_df = pd.DataFrame.from_records(logs)
    if os.path.exists(log_file) and len(''.join([line for line in open(log_file, "r")]).strip()):
        log_df = pd.concat([pd.read_csv(log_file), log_df])
    log_df.to_csv(log_file, index=False)
    print("Done.")


if __name__=="__main__":
    #generate_statements(num_agents=5, model="gpt-4o-mini")

    #TODO
    # - Move script files to new folder scripts
    # - Add some tests as well
    from generative_social_choice.statements.partitioning import BaselineEmbedding
    agents = get_simple_agents()
    #print(BaselineEmbedding().compute(agents=agents))

    from generative_social_choice.statements.partitioning import KMeansClustering
    partitioning = KMeansClustering(embedding_method=BaselineEmbedding())
    print(partitioning.assign(agents=agents, num_partitions=3))