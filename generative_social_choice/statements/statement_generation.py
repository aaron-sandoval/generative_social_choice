import random
import string
import json
from dataclasses import dataclass
from typing import override, Tuple, List

import pandas as pd
import numpy as np

from generative_social_choice.queries.query_interface import Generator,Agent
from generative_social_choice.utils.gpt_wrapper import LLMLog, GPT
from generative_social_choice.queries.query_chatbot_personalization import ChatbotPersonalizationGenerator
from generative_social_choice.statements.partitioning import Partition


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


@dataclass
class GenerationResult:
    """Data class to manage resukts of statement generation.
    
    Note that other in addition to the statements we also want to include context information for logging."""

    statement: str
    generation_method: str
    agent_ids: list[str]


class NamedGenerator(Generator):
    """Interface class for generation methods
    
    Almost the same as Generator, but we want to ensure that the arguments passed to init
    can be obtained later for logging purposes."""
    _init_args: dict={}  # Remember init arguments for logging purposes

    def generate_with_context(self, agents: List[Agent]) -> Tuple[List[GenerationResult], List[LLMLog]]:
        """Same as generate() but returns additional context in the first part of the returned variable.
        
        This additional context is useful for logging in case the generator calls other generators."""
        statements, logs = self.generate(agents=agents)
        results = [GenerationResult(statement=statement, generation_method=self.name, agent_ids=sorted([agent.id for agent in agents]))
                   for statement in statements]
        return results, logs

    @property
    def name(self):
        return self.__class__.__name__ + "(" + ", ".join(f"{key}={value}" for key, value in self._init_args.items()) + ")"


class DummyGenerator(NamedGenerator):
    """Dummy method that returns random strings as new statements.
    
    Use for test purposes only!"""

    def __init__(self, num_statements: int=5, statement_length: int=20):
        self.num_statements = num_statements
        self.statement_length = statement_length
        self._init_args = {"num_statements": num_statements, "statement_length": statement_length}

    def generate(self, agents: List[Agent]) -> Tuple[List[str], List[LLMLog]]:
        """
        Returns random strings of fixed length with letters and whitespace.
        """
        statements = []
        for _ in range(self.num_statements):
            new_statement = ''.join(random.choices(string.ascii_letters + " ", k=self.statement_length))
            statements.append(new_statement)
        return statements, []
    

class NamedChatbotPersonalizationGenerator(ChatbotPersonalizationGenerator, NamedGenerator):
    """Use an LLM to generate a new statement for the given agents.
    
    This class is very similar to queries.query_chatbot_personalization.ChatbotPersonalizationGenerator,
    but has the additional name property for logging"""
    def __init__(
        self,
        *,
        seed: int | None = None,
        gpt_temperature=0.0,
        model: str = "gpt-4o-mini-2024-07-18",
    ):
        init_args = {"seed": seed, "gpt_temperature": gpt_temperature, "model": model}
        super().__init__(**init_args)
        self._init_args = init_args


class LLMGenerator(NamedGenerator):
    """Own implementation of statement generation with LLMs.
    
    Can generate multiple statements for the given agents.
    
    Useful as a baseline method."""
    def __init__(
        self,
        *,
        seed: int | None = None,
        gpt_temperature=0.0,
        model: str = "gpt-4o-mini-2024-07-18",
        num_statements: int=5,
    ):
        self.info = f"(seed={seed}, gpt_temperature={gpt_temperature})"
        self.num_statements = num_statements

        if seed is None:
            self.random = random  # inherit global randomness
        else:
            self.random = random.Random(seed)

        # Notable changes to the prompt in ChatbotPersonalizationGenerator:
        # - Not mentioning that users are organized into subgroups (which I found somewhat confusing; also doesn't apply the same way in our case)
        # - Asking to generate several statements
        # - Changing the prompt layout, using a list for the conditions
        self.system_prompt = f"""In the following, I will show you a list of users and their opinions regarding chatbot personalization.
        
Identify the {self.num_statements} most salient opinions among the distinct views expressed by these users.
Return exactly {self.num_statements} statements in JSON format, using a key 'statements' which has a list of the generated statements as strings.

Importantly, each statement has to satisfy the following conditions:
- The statement is ADVOCATING FOR A SINGLE SPECIFIC VIEW ONLY, NOT A SUMMARY OF ALL VIEWS.
- The statement starts with 'The most important rule for chatbot personalization is' and then GIVES A SINGLE, CONCRETE RULE.
- Then, in a second point, the statement provides a justification why this is the most important rule.
- Then, include a CONCRETE example of why this rule would be beneficial.
- Each statement should be no more than 50 words."""

        self.gpt_params = {
            "model": model,
            "temperature": gpt_temperature,
            "max_tokens": 500,
            "top_p": 1,  # Does this make sense for temperature>0?
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
        }

        self.llm = GPT(**self.gpt_params)

        self._init_args = {"seed": seed, "temperature": gpt_temperature, "model": model, "num_statements": num_statements}

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

        df = agent_summaries.copy()
        df = df.rename(columns={"summary": "statement"})
        
        df = df.to_json(orient="records")

        response, completion, log = self.llm.call(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": df},
            ],
            response_format={ "type": "json_object" },
        )
        statements = json.loads(response)["statements"]

        return statements, [{**log, "query_type": "generative"}]


class PartitionGenerator(NamedGenerator):
    """Nested method which takes a partitioning method and basic generator,
    and generates statements by partitioning all agents and running the basic generator
    on each partition."""

    def __init__(self, partitioning: Partition, base_generator: NamedGenerator, num_partitions: int):
        self.partitioning = partitioning
        self.base_generator = base_generator
        self.num_partitions = num_partitions

    @override
    def generate_with_context(self, agents: List[SimplePersonalizationAgent]) -> Tuple[List[GenerationResult], List[LLMLog]]:
        partitions = self.partitioning.assign(agents=agents, num_partitions=self.num_partitions)

        results = []
        logs = []
        for i in range(self.num_partitions):
            assigned_agent_ixs = np.where(partitions==i)[0]
            if len(assigned_agent_ixs)<1:
                continue

            assigned_agents = [agent for ix, agent in enumerate(agents) if ix in assigned_agent_ixs]

            result, lgs = self.base_generator.generate_with_context(agents=assigned_agents)
            results.extend(result)
            logs.extend(lgs)
        return results, logs
    
    def generate(self, agents: List[SimplePersonalizationAgent]) -> Tuple[List[str], List[LLMLog]]:
        results, logs = self.generate_with_context(agents=agents)
        statements = [result.statement for result in results]
        return statements, logs