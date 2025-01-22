import os
import random
import pandas as pd

from typing import Optional

from generative_social_choice.utils.helper_functions import (
    get_base_dir_path,
    get_time_string,
)
from generative_social_choice.statements.partitioning import BaselineEmbedding
from generative_social_choice.statements.partitioning import KMeansClustering
from generative_social_choice.statements.statement_generation import (
    get_simple_agents,
    DummyGenerator,
    NamedChatbotPersonalizationGenerator,
    LLMGenerator,
    PartitionGenerator,
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


def generate_statements(num_agents: Optional[int] = None, model: str = "default", debug_mode: bool=False):
    gen_query_model_arg = {"model": model} if model != "default" else {}

    # Set up agents
    agents = get_simple_agents()

    # Subsample agents
    if num_agents is not None:
        agents = random.sample(agents, num_agents)

    # Set up generators

    #TODO Our generators don't use seeds yet
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
        PartitionGenerator(
            partitioning=KMeansClustering(embedding_method=BaselineEmbedding(), num_partitions=3),
            base_generator=DummyGenerator(num_statements=2),
        ),
    ]

    # Now for all the generators, generate statements, then write the results to some file
    results = []
    logs = []
    for generator in generators:
        r, lgs = generator.generate_with_context(agents=agents)
        results.extend([result.to_dict() for result in r])
        logs.extend(lgs)
    
    # Output into data/demo_data with timestring prepended
    print("Writing results to file ...")
    timestring = get_time_string()
    dirname = (
        get_base_dir_path()
        / "data/demo_data"
        / (f"{timestring}_statement_generation" if not debug_mode else "TEST_statement_generation")
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
    # NOTE For running this, you need to have the package installed
    # (`pip install -e .` from the folder where README.md is located)
    generate_statements(num_agents=5, model="gpt-4o-mini", debug_mode=True)

    #TODO
    # - Add option to save embeddings and load from file