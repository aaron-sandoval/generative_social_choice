from generative_social_choice.utils.helper_functions import (
    get_base_dir_path,
    get_time_string,
)
import pandas as pd
from generative_social_choice.queries.query_chatbot_personalization import (
    ChatbotPersonalizationAgent,
    ChatbotPersonalizationGenerator,
    SubsamplingChatbotPersonalizationGenerator,
    NearestNeighborChatbotPersonalizationGenerator,
    find_nearest_neighbors_by_embeddings,
    find_nearest_neighbors,
)
from generative_social_choice.slates.slate_generation import (
    generate_slate_ensemble_greedy,
)
import random
import os
import time
import argparse
import concurrent.futures
from typing import Optional, Literal


def get_results_dir(embedding_type: Literal["llm", "seed_statement"], candidates_persist: bool):
    """Directory holding the runs of one configuration. Runs are identified by their position among
    the sorted run directories, so each configuration needs its own directory."""
    results_dir = "generate_slate_results_openai_embeddings" if embedding_type == "seed_statement" else "generate_slate_results_baseline"
    if candidates_persist:
        results_dir += "_persistent_pool"
    return get_base_dir_path() / "data" / "demo_data" / results_dir


def generate_slate_from_paper(
    num_agents: Optional[int],
    generator_model: str = "default",
    discriminator_model: str = "default",
    dir_suffix: Optional[str] = "from_paper",
    embedding_type: Literal["llm", "seed_statement"] = "llm",
    candidates_persist: bool = False,
    num_threads: int = 1,
):
    if embedding_type == "seed_statement":
        nn_function = find_nearest_neighbors_by_embeddings
    elif embedding_type == "llm":
        nn_function = find_nearest_neighbors
    else:
        raise ValueError(f"Invalid embedding_type: {embedding_type}")

    disc_query_model_arg = {"model": "gpt-4o-mini-2024-07-18"} if discriminator_model == "default" else {"model": discriminator_model}
    # The earlier runs used "gpt-4o", which resolved to this snapshot (see the model in their logs)
    gen_query_model_arg = {"model": "gpt-4o-2024-08-06"} if generator_model == "default" else {"model": generator_model}

    # Set up agents

    df = pd.read_csv(get_base_dir_path() / "data/chatbot_personalization_data.csv")
    df = df[df["sample_type"] == "generation"]
    agent_id_to_summary = (
        pd.read_csv(get_base_dir_path() / "data/user_summaries_generation.csv")
        .set_index("user_id")["summary"]
        .to_dict()
    )
    # The data are stored with LF line endings, but a Windows checkout (core.autocrlf) turns newlines
    # inside CSV fields into CRLF. That changes the prompts and shifts ratings by about 0.1, so always
    # use LF, as in the GenerativeCC runs (see data/model_drift/RESULT_model_drift.md).
    to_lf = lambda x: x.replace("\r\n", "\n") if isinstance(x, str) else x
    df = df.apply(lambda col: col.map(to_lf))
    agent_id_to_summary = {agent_id: to_lf(summary) for agent_id, summary in agent_id_to_summary.items()}

    agents = []
    for id in df.user_id.unique():
        agent = ChatbotPersonalizationAgent(
            id=id,
            survey_responses=df[df.user_id == id],
            summary=agent_id_to_summary[id],
            **disc_query_model_arg,
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
        ChatbotPersonalizationGenerator(
            seed=0, gpt_temperature=0, **gen_query_model_arg
        ),
        ChatbotPersonalizationGenerator(
            seed=0, gpt_temperature=1, **gen_query_model_arg
        ),
        SubsamplingChatbotPersonalizationGenerator(
            sample_size=5, seed=1, gpt_temperature=0, **gen_query_model_arg
        ),
        # Note: seeds are None for NN generators because their
        # behavior was determined by the global seed
        NearestNeighborChatbotPersonalizationGenerator(
            sample_size=20,
            nbhd_size=5,
            seed=None,
            gpt_temperature=0,
            nn_function=nn_function,
            **gen_query_model_arg,
        ),
        NearestNeighborChatbotPersonalizationGenerator(
            sample_size=100,
            nbhd_size=5,
            seed=None,
            gpt_temperature=0,
            nn_function=nn_function,
            **gen_query_model_arg,
        ),
        NearestNeighborChatbotPersonalizationGenerator(
            sample_size=100,
            nbhd_size=10,
            seed=None,
            gpt_temperature=0,
            nn_function=nn_function,
            **gen_query_model_arg,
        ),
    ]

    # Generate slate

    results_dir = get_results_dir(embedding_type, candidates_persist)
    results_dir.mkdir(parents=True, exist_ok=True)
    # Runs in parallel processes can start in the same second, so retry with a later timestamp
    while True:
        full_log_dir = results_dir / f"{get_time_string()}__generate_slate_{dir_suffix}"
        try:
            os.mkdir(full_log_dir)
            break
        except FileExistsError:
            time.sleep(1)

    print(
        f"Generating slate for {len(agents)} agents with {len(generators)} generators"
    )

    slate, agents_round_matched, slate_utilities = generate_slate_ensemble_greedy(
        agents=agents,
        generators=generators,
        slate_size=5,
        candidates_persist=candidates_persist,
        full_log_dir=full_log_dir,
        verbose=True,
        num_threads=num_threads,
    )

    return slate, agents_round_matched, slate_utilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_agents",
        type=int,
        default=None,
        help="Number of agents to consider when generating slate. If not provided, all agents will be used.",
    )

    parser.add_argument(
        "--generator_model",
        type=str,
        default="default",
        help="Specify a model to use for statement generation. Default is gpt-4o-2024-08-06. Fish's experiments (late 2023) used gpt-4-base (publicly unavailable) for the discriminative queries and gpt-4-32k-0613 for the generative queries.",
    )

    parser.add_argument(
        "--discriminator_model",
        type=str,
        default="default",
        help="Specify a model to use for statement generation. Default is gpt-4o-mini-2024-07-18. Fish's experiments (late 2023) used gpt-4-base (publicly unavailable) for the discriminative queries and gpt-4-32k-0613 for the generative queries.",
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="llm",
        choices=["llm", "seed_statement"],
        help="How the nearest-neighbor generators find neighbors: 'llm' uses discriminative queries (as in Fish et al.), 'seed_statement' uses precomputed OpenAI embeddings.",
    )

    parser.add_argument(
        "--candidates_persist",
        action="store_true",
        help="Keep the candidates of earlier rounds in the pool, as described in Fish et al. (Section 4.2, footnote 13). Their released code defaults to only considering the current round's candidates.",
    )

    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of runs to execute in parallel. Each run is serial, like the earlier runs: with threads, the generators would share the global random state in a nondeterministic order.",
    )

    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Threads for LLM calls within a run. Also parallelizes the generators, see --num_processes.",
    )
    args = parser.parse_args()
    dir_suffix = {"llm": "via_fish_nn", "seed_statement": "via_openai_embeddings_nn"}[args.embedding_type]
    run_args = dict(
        num_agents=args.num_agents, generator_model=args.generator_model, discriminator_model=args.discriminator_model,
        dir_suffix=dir_suffix, embedding_type=args.embedding_type, candidates_persist=args.candidates_persist, num_threads=args.num_threads,
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_processes) as executor:
        futures = []
        for _ in range(args.num_runs):
            futures.append(executor.submit(generate_slate_from_paper, **run_args))
            # Stagger the starts so that the run directories' timestamps differ
            time.sleep(2)
        for future in concurrent.futures.as_completed(futures):
            future.result()
