import math
import unittest

import pandas as pd
import random

from gen_social_choice.queries.query_chatbot_personalization import (
    ChatbotPersonalizationAgent,
    ChatbotPersonalizationGenerator,
)
from gen_social_choice.slates.slate_generation import (
    generate_slate_ensemble_greedy,
)
from gen_social_choice.utils.helper_functions import get_base_dir_path


class TestGenerateSlate(unittest.TestCase):
    def generate_slate_test(
        self,
        num_agents: int,
        slate_size: int,
        disc_query_model: str,
        gen_query_model: str,
    ):
        disc_query_model_arg = (
            {"model": disc_query_model} if disc_query_model != "default" else {}
        )
        gen_query_model_arg = (
            {"model": gen_query_model} if gen_query_model != "default" else {}
        )

        random.seed(0)

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
        agents = random.sample(agents, k=num_agents)

        generators = [
            ChatbotPersonalizationGenerator(
                seed=0, gpt_temperature=0.0, **gen_query_model_arg
            ),
        ]

        slate, agents_round_matched, slate_utilities = generate_slate_ensemble_greedy(
            agents=agents,
            generators=generators,
            slate_size=slate_size,
        )

        # Check slate has expected size
        self.assertEqual(len(slate), slate_size)

        # Each candidate should have n//k or ceil(n/k) agents assigned
        coalition_sizes = [
            len(
                [
                    agent_id
                    for agent_id in agents_round_matched
                    if agents_round_matched[agent_id] == round_num
                ]
            )
            for round_num in set(agents_round_matched.values())
        ]
        self.assertTrue(
            all(
                coalition_size == len(agents) // slate_size
                or coalition_size == math.ceil(len(agents) / slate_size)
                for coalition_size in coalition_sizes
            )
        )

    def test_generate_slate_small_replication(self):
        self.generate_slate_test(3, 2, "default", "default")

    def test_generate_slate_small_fast(self):
        self.generate_slate_test(3, 2, "gpt-4o-mini", "gpt-4o-mini")

    def test_generate_slate_large_fast(self):
        self.generate_slate_test(50, 5, "gpt-4o-mini", "gpt-4o-mini")
