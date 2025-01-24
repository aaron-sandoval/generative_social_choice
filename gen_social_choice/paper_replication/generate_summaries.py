import os
import pandas as pd
from generative_social_choice.utils.dataframe_completion import DataFrameCompleter
from generative_social_choice.utils.gpt_wrapper import GPT
from generative_social_choice.utils.helper_functions import (
    get_base_dir_path,
    get_time_string,
)
from tqdm import tqdm
from typing import Optional
import argparse


def generate_summaries(num_agents: Optional[int] = None, model: str = "default"):
    # Load survey response data and format df for summarization (using df completion tool)

    CUMULATIVE_SUMMARY_COL_NAME = """cumulative summary of all responses of this user so far, in the following structure:

    Pros and cons of chatbot personalization:

    Concrete illustrative examples:

    Proposed rules to regulate chatbot personalization:

    Reasoning behind rules:"""

    df = pd.read_csv(get_base_dir_path() / "data/chatbot_personalization_data.csv")

    df = df[
        df["question_content"].isin(
            ["trade-offs opinion", "rules opinion", "convince opinion"]
        )
    ][["user_id", "question_text", "text"]]
    df.rename({"question_text": "question", "text": "response"}, axis=1, inplace=True)
    df["summary of this response"] = "LLM_TODO"
    df[CUMULATIVE_SUMMARY_COL_NAME] = "LLM_TODO"
    df.set_index("user_id", inplace=True)

    # Set up prompt for df completer

    gpt_4_params = {
        "model": model,
        "temperature": 1,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
    }

    system_prompt = "Your task is to summarize survey responses as CONCISELY as possible. No need for formal language or nice phrasing, bullet points are fine. It should be as easy to parse as possible. DO NOT use your knowledge about the world, stick to what the participants said. Do not put quotes around your response."

    llm = GPT(**gpt_4_params)
    completer = DataFrameCompleter(llm=llm)

    # Subsample users

    if num_agents is not None:
        user_ids = list(df.index.unique())
        user_ids = user_ids[:num_agents]
    else:
        user_ids = list(df.index.unique())

    # Generate summaries of chosen users

    for user_id in user_ids[:5]:
        n_todos = (df.loc[user_id] == completer.todo_marker).sum().sum()

        log = []
        for _ in tqdm(range(n_todos)):
            df.loc[user_id], log_line = completer.complete(
                df.loc[user_id], system_prompt=system_prompt, verbose=False
            )
            log += [log_line]

        n_todos = (df.loc[user_id] == completer.todo_marker).sum().sum()
        assert n_todos == 0

    # Save summaries

    timestring = get_time_string()
    dirname = (
        get_base_dir_path()
        / "data/demo_data"
        / f"{timestring}_user_summaries_generation"
    )
    os.makedirs(dirname)

    df.to_csv(dirname / "user_summaries_generation_raw_output.csv")

    final_question = "Your Opinion\nSuppose you had to convince others of your proposed rules, what would be your strongest arguments?"

    df[df["question"] == final_question].rename(
        {CUMULATIVE_SUMMARY_COL_NAME: "summary"}, axis=1
    )["summary"].str.replace("LLM_DONE\n", "").to_csv(
        dirname / "user_summaries_generation.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_agents",
        type=int,
        default=None,
        help="Number of agents to generate summaries for. If not provided, all agents will be used.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="default",
        help="Default is gpt-4o-mini-2024-07-18. Fish's experiments (late 2023) used gpt-4-32k-0613 (publicly unavailable).",
    )

    args = parser.parse_args()

    generate_summaries(num_agents=args.num_agents, model=args.model)
