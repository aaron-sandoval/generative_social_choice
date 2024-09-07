from string import Template
from typing import List, Tuple
import pandas as pd
import random

from chatbot_personalization.utils.gpt_wrapper import GPT
from chatbot_personalization.utils.dataframe_completion import DataFrameCompleter
from chatbot_personalization.queries.query_interface import Agent, Generator, LLMLog
from chatbot_personalization.utils.gpt_wrapper import get_probabilities_from_completion


##################################################
# Discriminative queries
##################################################


def generate_fewshot_prompt_template(
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
    question = "$statement"  # Note: no question prepended here
    few_shot_dict[question] = {
        "choices": list(approval_levels.keys()),
        "choice_numbers": list(approval_levels.values()),
        "choice_number": None,
    }

    few_shot_prompt = str(few_shot_dict)
    few_shot_prompt = few_shot_prompt[: few_shot_prompt.rfind(" None}}")]
    # Last token of prompt will be ":". We expect LLM to complete " <choice number>", with a space first.
    # Prompt is designed this way due to weird tokenization quirks (at the time it was bad to end prompt with space).

    return f"""{{"FREEFORM_RESPONSES": {freeform_responses}, "RATING_RESPONSES": {few_shot_prompt}"""


class ChatbotPersonalizationAgent(Agent):
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
    ):
        self.id = id
        self.survey_responses = survey_responses
        self.summary = summary

        self.prompt_template = generate_fewshot_prompt_template(
            survey_responses=survey_responses,
            approval_levels=ChatbotPersonalizationAgent.approval_levels,
        )

        if model == "gpt-4-base":
            ## This is what was used in the paper! ##
            self.gpt_params = {
                "model": "gpt-4-base",
                "temperature": 0,
                "logprobs": 10,
                "stop": [",", "}"],
                "max_tokens": 2,
            }
            self.token_idx = 1  # due to legacy model quirks, LLM writes space first, then approval level
        else:  # e.g. using gpt-4o
            ## This was not used in the paper -- it's here to make running this code easier in case gpt-4-base is inaccessible or too expensive ##
            self.gpt_params = {
                "model": model,
                "temperature": 0,
                "logprobs": True,
                "top_logprobs": 10,
                "stop": [",", "}"],
                "max_tokens": 1,
            }
            self.token_idx = 0  # LLM immediately writes approval level

        self.llm = GPT(**self.gpt_params)

    def get_id(self):
        return self.id

    def get_description(self):
        return self.summary

    def get_approval(
        self, statement: str, use_logprobs: bool = True
    ) -> tuple[float, list[LLMLog]]:
        prompt = Template(self.prompt_template).safe_substitute(statement=statement)

        _, completion, log = self.llm.call(prompt=prompt)

        # Compute expected approval from log probs
        probs = get_probabilities_from_completion(
            completion=completion, token_idx=self.token_idx
        )
        probs = probs[pd.to_numeric(probs.index, errors="coerce").notna().astype(bool)]

        if not use_logprobs:
            return float(probs.idxmax()), [{**log, "query_type": "discriminative"}]

        expected_approval = float((probs * probs.index.astype(float)).sum())

        log["response"] = (
            expected_approval  # overwrite token completed with expected approval
        )

        # We don't normalize because it wouldn't make a difference -- LLM
        # essentially always outputs valid approval level
        return expected_approval, [{**log, "query_type": "discriminative"}]


##################################################
# Generative queries
##################################################


class ChatbotPersonalizationGenerator(Generator):
    def __init__(
        self,
        *,
        seed: int | None = None,
        gpt_temperature=0.0,
        model: str = "gpt-4-32k-0613",
    ):
        self.info = f"(seed={seed}, gpt_temperature={gpt_temperature})"

        if seed is None:
            self.random = random  # inherit global randomness
        else:
            self.random = random.Random(seed)

        self.system_prompt = "In the following, I will show you a list of users and their opinions regarding chatbot personalization. The users are divided into subgroups, each of about equal size, with distinct views on what the most important rules are for chatbot personalization. Identify the most salient one among these distinct views. Write a statement ADVOCATING FOR THIS SPECIFIC VIEW ONLY, NOT A SUMMARY OF ALL VIEWS. Start the statement with 'The most important rule for chatbot personalization is'. GIVE A SINGLE, CONCRETE RULE. Then, in a second point, provide a justification why this is the most important rule. Then, give an CONCRETE example of why this rule would be beneficial. Write no more than 50 words."

        self.gpt_params = {
            "model": model,
            "temperature": gpt_temperature,
            "max_tokens": 500,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
        }

        llm = GPT(**self.gpt_params)
        self.completer = DataFrameCompleter(llm=llm, done_marker="")

    def generate(self, agents: list[Agent]) -> tuple[list[str], list[LLMLog]]:
        # Determine random order to shuffle agents
        # (for reproducibility, using random.shuffle specfically)
        agent_ids = [agent.get_id() for agent in agents]
        self.random.shuffle(agent_ids)
        # Create df with agent summaries
        agent_id_to_summary = {
            agent.get_id(): agent.get_description() for agent in agents
        }
        agent_summaries = pd.DataFrame(
            [
                {"user_id": agent_id, "summary": agent_id_to_summary[agent_id]}
                for agent_id in agent_ids
            ]
        ).set_index("user_id")

        df_to_complete = agent_summaries.copy()
        df_to_complete = df_to_complete.rename(columns={"summary": "statement"})
        # Add extra row at bottom where LLM will write generated statement
        df_to_complete.loc["subgroup"] = self.completer.todo_marker
        df_to_complete = df_to_complete.reset_index()

        # Make LLM query to generate statement
        completed, log = self.completer.complete(
            df_to_complete, system_prompt=self.system_prompt, verbose=False
        )

        completed = completed.set_index("user_id")
        # LLM will have written statement in next available cell, which is
        # row 'subgroup' and col 'statement' (only one col)
        plurality_opinion = completed.loc["subgroup", "statement"]
        return [plurality_opinion], [{**log, "query_type": "generative"}]


class SubsamplingChatbotPersonalizationGenerator(ChatbotPersonalizationGenerator):
    """
    Randomly subsample sample_size agents, and generate a statement using only those agents.
    """

    def __init__(
        self,
        *,
        sample_size: int,
        seed=0,
        gpt_temperature=0.0,
        model: str = "gpt-4-32k-0613",
    ):
        self.sample_size = sample_size
        super().__init__(seed=seed, gpt_temperature=gpt_temperature, model=model)

    def generate(self, agents: List[Agent]) -> Tuple[str, List[LLMLog]]:
        if len(agents) > self.sample_size:
            sampled_agents = self.random.sample(agents, self.sample_size)
        else:
            sampled_agents = agents.copy()
        return super().generate(sampled_agents)


def find_nearest_neighbors(
    center_agent: Agent, agents: List[Agent], nbhd_size: int
) -> Tuple[List[Agent], List[LLMLog]]:
    """
    Using approval queries, returns the nbhd_size nearest neighbors to the center_agent, among the list agents.
    """
    assert center_agent not in agents

    logs = []
    # Compute approval levels for all agents
    agent_to_approval_level = {}
    for agent in agents:
        approval, log = agent.get_approval(center_agent.get_description())
        agent_to_approval_level[agent] = approval
        logs.extend(log)

    # Sort agents by approval level (reverse because higher approval level=better)
    sorted_agents = sorted(
        agents, key=lambda agent: agent_to_approval_level[agent], reverse=True
    )

    # Find nearest neighbors
    cluster = set(sorted_agents[:nbhd_size])

    return cluster, logs


class NearestNeighborChatbotPersonalizationGenerator(ChatbotPersonalizationGenerator):
    """
    Subsample k agents. From this population, sample an agent (the "center agent"). Find the m 'nearest neighbors' to the center agent among the k agents, using the discriminative query, and then run the simple ChatbotPersonalizationGenerator on this set of m+1 agents. This will produce a statement.

    In the paper we run this with (k=20, m=5), (k=100, m=5), and (k=100, m=10).
    """

    def __init__(
        self,
        *,
        sample_size: int,
        nbhd_size: int,
        seed=0,
        gpt_temperature=0,
        model: str = "gpt-4-32k-0613",
    ):
        self.sample_size = sample_size
        self.nbhd_size = nbhd_size
        super().__init__(seed=seed, gpt_temperature=gpt_temperature, model=model)

    def generate(self, agents: List[Agent]) -> Tuple[str, List[LLMLog]]:
        logs = []
        # Subsample sample_size agents
        if len(agents) > self.sample_size:
            sampled_agents = self.random.sample(agents, self.sample_size)
        else:
            sampled_agents = agents.copy()

        # Find nbhd_size nearest neighbors of randomly sampled center_agent
        center_agent = self.random.choice(sampled_agents)
        sampled_agents.remove(center_agent)
        nn_agents, nn_logs = find_nearest_neighbors(
            center_agent, agents=sampled_agents, nbhd_size=self.nbhd_size
        )
        logs.extend(nn_logs)

        # Generate statement from nearest neighbors
        statement, gen_logs = super().generate(list(nn_agents) + [center_agent])
        logs.extend(gen_logs)

        return statement, logs
