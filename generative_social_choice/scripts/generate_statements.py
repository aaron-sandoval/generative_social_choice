import os
import random
import pandas as pd

from typing import Optional
from pathlib import Path

from generative_social_choice.utils.helper_functions import (
    get_base_dir_path,
    get_time_string,
)
from generative_social_choice.statements.partitioning import (
    BaselineEmbedding,
    KMeansClustering,
    PrecomputedEmbedding,
    PrecomputedPartition,
    Partition,
)
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


def generate_statements(num_agents: Optional[int] = None, model: str = "default", folder_name: Optional[str] = None,
                        partioning_file: Optional[Path] = None, partitioning: Optional[Partition] = None):
    gen_query_model_arg = {"model": model} if model != "default" else {}

    # Set up agents
    agents = get_simple_agents()

    if partioning_file is not None:
        if partioning_file.exists():
            print("Reading partitioning from existing file ...")
            partitioning = PrecomputedPartition(filepath=partioning_file)
        else:
            assert partitioning is not None, "If partitioning_file is given and the file doesn't exist, a partitioning method has to be given!"
            print("Precomputing partitioning and storing to file ...")
            partitioning.precompute(filepath=partioning_file, agents=agents)
            partitioning = PrecomputedPartition(filepath=partioning_file)

    # Subsample agents
    if num_agents is not None:
        agents = random.sample(agents, num_agents)

    # Set up generators

    generators = [
        #DummyGenerator(num_statements=3),
        NamedChatbotPersonalizationGenerator(
            seed=0, gpt_temperature=0, **gen_query_model_arg
        ),
        NamedChatbotPersonalizationGenerator(
            seed=0, gpt_temperature=1, **gen_query_model_arg
        ),
        LLMGenerator(
            seed=0, gpt_temperature=0, num_statements=5, **gen_query_model_arg
        ),
        LLMGenerator(
            seed=0, gpt_temperature=1, num_statements=5, **gen_query_model_arg
        ),
        #PartitionGenerator(
        #    partitioning=KMeansClustering(embedding_method=BaselineEmbedding(), num_partitions=3),
        #    base_generator=DummyGenerator(num_statements=2),
        #),
        PartitionGenerator(
            partitioning=partitioning,
            base_generator=NamedChatbotPersonalizationGenerator(seed=0, gpt_temperature=0, **gen_query_model_arg),
        ),
        PartitionGenerator(
            partitioning=partitioning,
            base_generator=NamedChatbotPersonalizationGenerator(seed=0, gpt_temperature=1, **gen_query_model_arg),
        ),
        PartitionGenerator(
            partitioning=partitioning,
            base_generator=LLMGenerator(seed=0, gpt_temperature=0, num_statements=3, **gen_query_model_arg),
        ),
        PartitionGenerator(
            partitioning=partitioning,
            base_generator=LLMGenerator(seed=0, gpt_temperature=1, num_statements=3, **gen_query_model_arg),
        ),
        #PartitionGenerator(
        #    partitioning=KMeansClustering(embedding_method=BaselineEmbedding(), num_partitions=3),
        #    base_generator=LLMGenerator(seed=0, gpt_temperature=0, num_statements=3, **gen_query_model_arg),
        #),
    ]
    generators = [generators[-1]]

    # Now for all the generators, generate statements, then write the results to some file
    results = []
    logs = []
    for generator in generators:
        print(f"Generating statements with generator {generator.name} ...")
        r, lgs = generator.generate_with_context(agents=agents)
        results.extend([result.to_dict() for result in r])
        logs.extend(lgs)
    
    # Output into data/demo_data with timestring prepended
    print("Writing results to file ...")
    timestring = get_time_string()
    dirname = (
        get_base_dir_path()
        / "data/demo_data"
        / (folder_name if folder_name is not None else f"{timestring}_statement_generation")
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
    #generate_statements(num_agents=5, model="gpt-4o-mini", folder_name="TEST_statement_generation")
    #generate_statements(model="gpt-4o-mini")

    # We want to precompute partitioning to use the same clustering for
    # different LLM generation methods
    # (This allows for comparing different LLM methods, but we can also try doing this differently)
    generate_statements(
        model="gpt-4o-mini",
        partioning_file=get_base_dir_path() / "data/demo_data/kmeans_partitioning_5.json",
        partitioning=KMeansClustering(embedding_method=BaselineEmbedding(), num_partitions=5, seed=0),
    )

    # How to use precomputed embeddings
    #embedding_file = get_base_dir_path() / "data/demo_data/TEST_embeddings.json"
    #BaselineEmbedding().precompute(agents=get_simple_agents(), filepath=embedding_file)
    #print("Computing embeddings and saving them to disk ...")

    # How to precompute assignments
    #partition_file = get_base_dir_path() / "data/demo_data/TEST_partitioning.json"
    #partitioning = KMeansClustering(num_partitions=5, embedding_method=BaselineEmbedding())
    #partitioning.precompute(agents=get_simple_agents(), filepath=partition_file)

    #print("Using precomputed embeddings for clustering")
    #partitioning = PrecomputedPartition(filepath=partition_file)
    #print(partitioning.assign(agents=agents))