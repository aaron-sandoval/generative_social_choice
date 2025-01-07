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
)
from generative_social_choice.slates.slate_generation import (
    generate_slate_ensemble_greedy,
)
import random
import os
import argparse
from typing import Optional


def generate_slate_from_paper(num_agents: Optional[int], model: str = "default"):
    disc_query_model_arg = {"model": model} if model != "default" else {}
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
            **gen_query_model_arg,
        ),
        NearestNeighborChatbotPersonalizationGenerator(
            sample_size=100,
            nbhd_size=5,
            seed=None,
            gpt_temperature=0,
            **gen_query_model_arg,
        ),
        NearestNeighborChatbotPersonalizationGenerator(
            sample_size=100,
            nbhd_size=10,
            seed=None,
            gpt_temperature=0,
            **gen_query_model_arg,
        ),
    ]

    # Generate slate

    full_log_dir = (
        get_base_dir_path()
        / "data"
        / "demo_data"
        / f"{get_time_string()}__generate_slate_from_paper"
    )
    os.mkdir(full_log_dir)

    print(
        f"Generating slate for {len(agents)} agents with {len(generators)} generators"
    )

    slate, agents_round_matched, slate_utilities = generate_slate_ensemble_greedy(
        agents=agents,
        generators=generators,
        slate_size=5,
        full_log_dir=full_log_dir,
        verbose=True,
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
        "--model",
        type=str,
        default="default",
        help="Specify a model to use for all queries. Default is gpt-4o-mini-2024-07-18. Fish's experiments (late 2023) used gpt-4-base (publicly unavailable) for the discriminative queries and gpt-4-32k-0613 for the generative queries.",
    )

    args = parser.parse_args()

    generate_slate_from_paper(num_agents=args.num_agents, model=args.model)
