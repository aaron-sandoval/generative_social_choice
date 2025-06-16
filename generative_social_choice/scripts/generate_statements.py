import os
import random
import pandas as pd
import argparse

from typing import Optional
from pathlib import Path

from generative_social_choice.utils.helper_functions import (
    get_base_dir_path,
    get_time_string,
    get_results_paths,
)
from generative_social_choice.statements.partitioning import (
    BaselineEmbedding,
    KMeansClustering,
    OpenAIEmbedding,
    PrecomputedPartition,
    Partition,
)
from generative_social_choice.statements.statement_generation import (
    get_simple_agents,
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


def generate_statements(num_agents: Optional[int] = None, model: str = "default", folder_name: Optional[str] = None, folder_path: Optional[Path] = None,
                        partioning_file: Optional[Path] = None, partitioning: Optional[Partition] = None, seed: int = 0):
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
        # We generate two statements meant to represent all agents
        NamedChatbotPersonalizationGenerator(
            seed=seed, gpt_temperature=1, **gen_query_model_arg
        ),
        NamedChatbotPersonalizationGenerator(
            seed=seed, gpt_temperature=1, **gen_query_model_arg
        ),
        LLMGenerator(
            seed=seed, gpt_temperature=1, num_statements=5, **gen_query_model_arg
        ),
        PartitionGenerator(
            partitioning=partitioning,
            base_generator=NamedChatbotPersonalizationGenerator(seed=seed, gpt_temperature=1, **gen_query_model_arg),
        ),
        PartitionGenerator(
            partitioning=partitioning,
            base_generator=NamedChatbotPersonalizationGenerator(seed=seed, gpt_temperature=1, **gen_query_model_arg),
        ),
        PartitionGenerator(
            partitioning=partitioning,
            base_generator=LLMGenerator(seed=seed, gpt_temperature=1, num_statements=5, **gen_query_model_arg),
        ),
    ]

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
    if folder_path is None:
        folder_path = get_base_dir_path() / "data/demo_data" / (folder_name if folder_name is not None else f"{get_time_string()}_statement_generation")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    #TODO Could get result_file and log_file paths from results_paths
    result_file = folder_path / "statement_generation_raw_output.csv"
    # NOTE: Pandas can write csvs in append mode, but that is error prone as previous header entries
    # aren't checked.
    result_df = pd.DataFrame.from_records(results)
    # Existing but empty files also create issues, so we check for that
    if os.path.exists(result_file) and len(''.join([line for line in open(result_file, "r")]).strip()):
        result_df = pd.concat([pd.read_csv(result_file), result_df])
    result_df.to_csv(result_file, index=False)

    log_file = folder_path / "statement_generation_logs.csv"
    log_df = pd.DataFrame.from_records(logs)
    if os.path.exists(log_file) and len(''.join([line for line in open(log_file, "r")]).strip()):
        log_df = pd.concat([pd.read_csv(log_file), log_df])
    log_df.to_csv(log_file, index=False)
    print("Statement generation done.")


def run(embedding_method: str, num_agents: int, num_clusters: int, model: str, seed: int, run_id: str | None = None):
    if embedding_method == "llm":
        embeddings = OpenAIEmbedding(model="text-embedding-3-small", use_summary=False)
    elif embedding_method == "seed_statement":
        embeddings = BaselineEmbedding()
    else:
        raise ValueError(f"Invalid embedding method: {embedding_method} (should be 'llm' or 'seed_statement')")

    # Note that labelling model isn't relevant for base_dir
    results_paths = get_results_paths(run_id=run_id, embedding_type=embedding_method, generation_model=model, labelling_model="4o-mini")
    base_dir = results_paths["base_dir"]
    partitioning_file = base_dir / f"kmeans_partitioning_{embedding_method}_{num_clusters}_{seed}.json"
    folder_path = base_dir / f"generated_with_{model}_using_{embedding_method}_embeddings"
    assert folder_path == results_paths["results_dir"], "Folder path and results path are not the same"

    if not os.path.exists(folder_path):
        print(f"Creating directory {folder_path} ...")
        os.makedirs(folder_path)

    # We want to precompute partitioning to use the same clustering for
    # different LLM generation methods
    # (This allows for comparing different LLM methods, but this could also be done differently)
    generate_statements(
        model=model,
        num_agents=num_agents,
        seed=seed,
        partioning_file=partitioning_file,
        partitioning=KMeansClustering(embedding_method=embeddings, num_partitions=num_clusters, seed=seed),
        folder_path=folder_path,
    )



if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_agents",
        type=int,
        default=None,
        help="Number of agents to generate statements for. If not provided, all agents will be used.",
    )

    parser.add_argument(
        "--num_clusters",
        type=int,
        default=5,
        help="Number of clusters to use in partitioning methods.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="4o-mini",
        help="Default is 4o-mini. Fish's experiments (late 2023) used gpt-4-32k-0613 (publicly unavailable).",
    )

    parser.add_argument(
        "--embeddings",
        type=str,
        default="llm",
        choices=["llm", "seed_statement"],
        help="Embedding method to use for partitioning.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for random number generator.",
    )

    args = parser.parse_args()

    run(embedding_method=args.embeddings, num_agents=args.num_agents, num_clusters=args.num_clusters, model=args.model, seed=args.seed)