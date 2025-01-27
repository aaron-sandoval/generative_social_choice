import random
import json
from typing import Optional, List

import pandas as pd

from generative_social_choice.queries.query_interface import Agent
from generative_social_choice.utils.gpt_wrapper import LLMLog, GPT
from generative_social_choice.utils.helper_functions import get_base_dir_path
from generative_social_choice.queries.query_chatbot_personalization import ChatbotPersonalizationAgent


# NOTE: THIS IMPLEMENTATION IS NOT FINISHED! SPECIFICALLY, THE RETURNED SCORES ARE NOT FINE-GRAINED
# BUT CLOSE TO INTEGER VALUES, AND FOR USING IT IN THE PIPELINE, OTHER RATING GENERATION WOULD
# HAVE TO BE ADJUSTED.

STATEMENTS_FILE = get_base_dir_path() / "data/demo_data/2025-01-27-112106_statement_generation/statement_generation_raw_output.csv"


def generate_fewshot_context(
    *, survey_responses: pd.DataFrame, approval_levels: dict[str, int]
) -> str:
    """
    Given survey data from single agent and approval levels, generate a few-shot prompt template for the discriminator.
    """

    # Collect responses to general opinion questions
    freeform_responses = survey_responses.loc[
        survey_responses["detailed_question_type"] == "general opinion"
    ]
    freeform_responses = freeform_responses.set_index("question_text_for_llm")["text"]
    freeform_responses = freeform_responses.to_dict()

    # Collect responses to rating statements (these are the fewshot examples)
    approval_responses = survey_responses[
        survey_responses["detailed_question_type"] == "rating statement"
    ].copy()
    approval_responses.rename(columns={"text": "explanation"}, inplace=True)
    approval_responses["question"] = approval_responses["question_text_for_llm"]
    approval_responses = approval_responses.set_index("question")[
        ["explanation", "choice"]
    ]
    approval_responses["choice_number"] = approval_responses["choice"].map(
        approval_levels
    )
    approval_responses = approval_responses.to_dict("index")

    few_shot_dict = dict()
    for question in approval_responses:
        response = approval_responses[question]
        few_shot_dict[question] = {
            "choices": list(approval_levels.keys()),
            "choice_numbers": list(approval_levels.values()),
            "choice_number": response["choice_number"],
            "choice": response["choice"],
            "explanation": response["explanation"],
        }

    return str({"FREEFORM_RESPONSES": freeform_responses, "RATING_RESPONSES": few_shot_dict})


class BatchChatbotPersonalizationAgent(ChatbotPersonalizationAgent):
    """Own implementation of the discriminative function using batch processing.
    """
    approval_levels = {
        "not at all": 0,
        "poorly": 1,
        "somewhat": 2,
        "mostly": 3,
        "perfectly": 4,
    }

    def __init__(
        self,
        *,
        id: str,
        survey_responses: pd.DataFrame,
        summary: str,
        model: str = "gpt-4-base",
        max_tokens: int = 5000,
    ):
        self.id = id
        self.survey_responses = survey_responses
        self.summary = summary

        self.system_prompt = f"""Below you find survey responses regarding chatbot personalization from a single user.
        
Your task is to predict approval ratings this user would assign to the list of statements given to you in the user prompt.
Give your answer in JSON format, using the following keys:

- 'statements': Reproduce the list of statements for which you predict the ratings
- 'ratings': List of numeric scores you predict the user would assign to the statements.
  The i-th entry corresponds to the i-th statements from the given list.

Each score has to be between {min(self.approval_levels.values())} and {max(self.approval_levels.values())}, where the numbers have the following meaning:

{'\n'.join([f"- {score}: {meaning}" for meaning, score in self.approval_levels.items()])}

Use float values for the scores to capture more nuanced differences.


###

SURVEY DATA

""" + generate_fewshot_context(survey_responses=survey_responses, approval_levels=self.approval_levels)
        
        ## This was not used in the paper -- it's here to make running this code easier in case gpt-4-base is inaccessible or too expensive ##
        self.gpt_params = {
            "model": model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
        }

        self.llm = GPT(**self.gpt_params)

    def get_approval(
            self, statement: str,
    ):
        return self.get_approvals(statement=statement)

    def get_approvals(
        self, statements: list[str],
    ) -> tuple[list[float], list[LLMLog]]:
        statement_dict = {"statements": statements}

        response, completion, log = self.llm.call(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": str(statement_dict)},
            ],
            response_format={ "type": "json_object" },
        )
        print(response)
        parsed_response = json.loads(response)
        ratings = parsed_response["ratings"]

        #TODO Rather order in correct way if needed
        for i, st in enumerate(parsed_response["statements"]):
            assert statements[i].lower().strip()==st.lower().strip()

        return ratings, [{**log, "query_type": "discriminative"}]


def get_agents(model: Optional[str] = None, **agent_kwargs) -> List[BatchChatbotPersonalizationAgent]:
    """Utility function to get all agents based on survey data and summaries"""
    if model is None:
        model = "gpt-4o-mini"
    
    df = pd.read_csv(get_base_dir_path() / "data/chatbot_personalization_data.csv")
    df = df[df["sample_type"] == "generation"]
    agent_id_to_summary = (
        pd.read_csv(get_base_dir_path() / "data/user_summaries_generation.csv")
        .set_index("user_id")["summary"]
        .to_dict()
    )

    agents = []
    for id in df.user_id.unique():
        agent = BatchChatbotPersonalizationAgent(
            id=id,
            survey_responses=df[df.user_id == id],
            summary=agent_id_to_summary[id],
            model=model,
            **agent_kwargs,
        )
        agents.append(agent)
    return agents

if __name__=="__main__":
    batch_size = 10
    agents = get_agents(model="gpt-4o-mini", max_tokens=batch_size * 300)
    #print(agents[0].system_prompt)

    statements = pd.read_csv(STATEMENTS_FILE)["statement"].to_list()

    result, logs = agents[0].get_approvals(statements=statements[:batch_size])
    print(result)