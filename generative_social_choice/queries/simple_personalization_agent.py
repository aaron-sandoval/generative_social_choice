##################################################
# Classes for statement generation
##################################################

from generative_social_choice.queries.query_interface import Agent, LLMLog


import pandas as pd


class SimplePersonalizationAgent(Agent):
    """Simple agent representation which doesn't require connecting to any LLM
    but can't be used to get approvals.

    We use this class in statement generation as computing new approvals
    is unnecessary."""

    def __init__(
        self,
        *,
        id: str,
        survey_responses: pd.DataFrame,
        summary: str,
    ):
        self.id = id
        self.survey_responses = survey_responses
        self.summary = summary

    def get_id(self):
        return self.id

    def get_description(self):
        return self.summary

    def get_approval(
        self, statement: str, use_logprobs: bool = True
    ) -> tuple[float, list[LLMLog]]:
        raise NotImplementedError()