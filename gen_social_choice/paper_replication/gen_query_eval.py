from itertools import product
import os
import argparse

from tqdm import tqdm

import random
import pandas as pd

from generative_social_choice.queries.query_chatbot_personalization import (
    ChatbotPersonalizationAgent,
    ChatbotPersonalizationGenerator,
    find_nearest_neighbors,
)
from generative_social_choice.utils.gpt_wrapper import GPT
from generative_social_choice.utils.helper_functions import (
    get_time_string,
    get_base_dir_path,
)

AGENT_TO_STATEMENT_PROMPT = """\
In a qualitative survey about chatbot personalization, a participant has given the following answers:
----
{desc}
----
Based on the opinions the user has articulated above, write a statement for them with the following, specific format: \
Start the statement with 'The most important rule for chatbot personalization is'. GIVE A SINGLE, CONCRETE RULE. \
Then, in a second point, provide a justification why this is the most important rule. \
Then, give an CONCRETE example of why this rule would be beneficial. Write no more than 50 words.
"""


def run(num_rounds: int, num_input_agents: int, model: str, seed: int):
    disc_query_model_arg = {"model": model} if model != "default" else {}
    gen_query_model_arg = {"model": model} if model != "default" else {}

    random.seed(seed)

    log_dir = (
        get_base_dir_path() / f"data/demo_data/{get_time_string()}__gen_query_eval"
    )
    os.mkdir(log_dir)
    logs_filename = f"{log_dir}/logs.csv"
    logs = []

    # for 'random 1', we generate a single statement for an agent
    # for some reason, 0314 was used here instead of 0613 -- difference is likely negligible
    gpt = GPT(model=model if model != "default" else "gpt-4-32k-0314")

    # Get agents

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

    ## TEMPORARY
    agents = random.sample(agents, 40)

    # Get moderator
    moderator = ChatbotPersonalizationGenerator(**gen_query_model_arg)

    # Run the experiment
    logs = []
    for round_num in range(num_rounds):
        print("=" * 80)
        print(f"Round {round_num+1}/{num_rounds}")
        print("=" * 80)
        # Get this round's input agents
        input_agents = random.sample(agents, num_input_agents)
        logs.append(
            {
                "round_num": round_num,
                "input_agents": [agent.get_id() for agent in input_agents],
            }
        )
        pd.DataFrame(logs).to_csv(logs_filename)
        ### Generate the 4 statements ##
        # Statement 1
        print("Generating statement 'random 1'...")
        statement1_agent = random.choice(input_agents)
        desc = statement1_agent.get_description()
        statement1, _, log = gpt.call(
            prompt=AGENT_TO_STATEMENT_PROMPT.format(desc=desc)
        )
        logs.append(
            {
                "round_num": round_num,
                "statement_type": "random_agent",
                "agent_id": statement1_agent.get_id(),
                "statement": statement1,
                **log,
            }
        )
        pd.DataFrame(logs).to_csv(logs_filename)
        # Statement 2
        print("Generating statement 'all'...")
        statement2, query1_logs2 = moderator.generate(agents=input_agents)
        assert len(query1_logs2) == 1
        assert len(statement2) == 1
        statement2 = statement2[0]
        logs.append(
            {
                "round_num": round_num,
                "statement_type": "all",
                "statement": statement2,
                **query1_logs2[0],
            }
        )
        pd.DataFrame(logs).to_csv(logs_filename)
        # Statement 3
        print("Generating statement 'random 5'...")
        statement3_agent_subset = random.sample(input_agents, 5)
        statement3, query1_logs3 = moderator.generate(agents=statement3_agent_subset)
        assert len(query1_logs3) == 1
        assert len(statement3) == 1
        statement3 = statement3[0]
        logs.append(
            {
                "round_num": round_num,
                "statement_type": "random_subset_5",
                "statement": statement3,
                "agent_subset": [agent.get_id() for agent in statement3_agent_subset],
                **query1_logs3[0],
            }
        )
        pd.DataFrame(logs).to_csv(logs_filename)
        # Statement 4
        print("Generating statement nn(s=5)...")
        statement4_center_agent = random.choice(input_agents)
        cluster, nn_logs4 = find_nearest_neighbors(
            statement4_center_agent,
            set(input_agents).difference({statement4_center_agent}),
            nbhd_size=5,
        )
        for d in nn_logs4:
            d = {"round_num": round_num, **d}
        logs.extend(nn_logs4)
        statement4_agents = list(cluster.union({statement4_center_agent}))
        statement4, query1_logs4 = moderator.generate(
            agents=statement4_agents,
        )
        assert len(query1_logs4) == 1
        assert len(statement4) == 1
        statement4 = statement4[0]
        logs.append(
            {
                "round_num": round_num,
                "statement_type": "nn_subset_6",
                "statement": statement4,
                "center_agent": statement4_center_agent.get_id(),
                "nn_agents": [agent.get_id() for agent in cluster],
                **query1_logs4[0],
            }
        )
        pd.DataFrame(logs).to_csv(logs_filename)

        # Evlauate statements
        statements = [statement1, statement2, statement3, statement4]
        statement_types = ["random 1", "all", "random 5", "nn(s=5)"]
        for agent, (statement, statement_type) in tqdm(
            product(input_agents, zip(statements, statement_types)),
            total=len(input_agents) * len(statements),
            desc=f"Evaluating {len(statements)} statements",
        ):
            query2_output, query2_logs = agent.get_approval(statement=statement)
            assert len(query2_logs) == 1
            logs.append(
                {
                    "round_num": round_num,
                    **query2_logs[0],
                    "query2_output": query2_output,
                    "agent_id": agent.get_id(),
                    "statement": statement,
                    "statement_type": statement_type,
                }
            )
            pd.DataFrame(logs).to_csv(logs_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_rounds",
        type=int,
        default=50,
        help="Number of times to repeat the experiment (1 time = 1 ensemble).",
    )
    parser.add_argument(
        "--num_input_agents",
        type=int,
        default=40,
        help="Number of 'input agents' to sample per round.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="Specify a model to use for all queries. Default is gpt-4o-mini-2024-07-18. Fish's experiments (late 2023) used gpt-4-base (publicly unavailable) for the discriminative queries and gpt-4-32k-0613 for the generative queries.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    run(
        num_rounds=args.num_rounds,
        num_input_agents=args.num_input_agents,
        model=args.model,
        seed=args.seed,
    )
