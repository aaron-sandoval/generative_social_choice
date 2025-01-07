import pandas as pd
import random

from generative_social_choice.queries.query_chatbot_personalization import (
    ChatbotPersonalizationAgent,
)
from generative_social_choice.utils.helper_functions import (
    get_base_dir_path,
    get_time_string,
)
from typing import Optional
from tqdm import tqdm
import os
import argparse


def validate_discriminative_query(
    *, num_samples: Optional[int] = None, model: str = "default"
):
    disc_query_model_arg = {"model": model} if model != "default" else {}

    ## Get agent-statement pairs

    df = pd.read_csv(get_base_dir_path() / "data/chatbot_personalization_data.csv")
    df = df[df["sample_type"] == "generation"]
    agent_id_to_summary = (
        pd.read_csv(get_base_dir_path() / "data/user_summaries_generation.csv")
        .set_index("user_id")["summary"]
        .to_dict()
    )
    agent_to_heldout_q = {}
    for id in df.user_id.unique():
        survey_responses = df[df.user_id == id]
        for heldout_q_idx in survey_responses[
            survey_responses["detailed_question_type"] == "rating statement"
        ].index:
            agent = ChatbotPersonalizationAgent(
                id=id,
                survey_responses=survey_responses.drop(heldout_q_idx),
                summary=agent_id_to_summary[id],
                **disc_query_model_arg,
            )
            agent_to_heldout_q[agent] = survey_responses.loc[heldout_q_idx].to_dict()

    ## Subsample queries to run

    if num_samples is not None:
        agent_to_heldout_q = dict(
            random.sample(list(agent_to_heldout_q.items()), num_samples)
        )

    ## Run queries

    logs = []
    log_dirname = (
        get_base_dir_path()
        / "data"
        / "demo_data"
        / f"{get_time_string()}__validate_disc_query"
    )
    os.mkdir(log_dirname)

    for agent, heldout_q in tqdm(agent_to_heldout_q.items()):
        statement = heldout_q["question_text_for_llm"]
        correct_choice = heldout_q["choice"]
        choice, log = agent.get_approval(statement)
        assert len(log) == 1
        logs.append(
            {**log[0], "correct_choice": correct_choice, "choice_number": choice}
        )
        pd.DataFrame(logs).to_csv(log_dirname / "logs.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of discriminative queries to validate. If not provided, all agent-statement pairs will be tested.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="If 'default', will use gpt-4-base, to exactly reproduce our experiments (subject to LLM stochasticity). Otherwise, will use the specified model.",
    )

    args = parser.parse_args()

    validate_discriminative_query(num_samples=args.num_samples, model=args.model)
