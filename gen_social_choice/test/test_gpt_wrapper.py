import unittest
from warnings import warn

from gen_social_choice.utils.gpt_wrapper import GPT


class TestGPT(unittest.TestCase):
    def test_completion_model_replication(self):
        warn(
            "gpt-4-base is publicly unavailable. This test is not applicable to the current version of the codebase."
        )

    def chat_model_test(self, model):
        params = {
            "model": model,
            "temperature": 0,
            "max_tokens": 50,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
        }
        llm = GPT(**params)
        prompt = "Who would win, an elephant or an ant?"

        response, _, _ = llm.call(
            messages=[
                {"role": "system", "content": "talk like jessie pinkman"},
                {"role": "user", "content": prompt},
            ]
        )

        self.assertTrue("yo" in response.lower() or "bitch" in response.lower())

    def test_chat_model_replication(self):
        warn(
            "gpt-4-32k-0613 is publicly unavailable. This test is not applicable to the current version of the codebase."
        )

    def test_chat_model_fast(self):
        self.chat_model_test(model="gpt-4o-mini")
