import unittest

from chatbot_personalization.utils.gpt_wrapper import GPT


class TestGPT(unittest.TestCase):
    def test_completion_model_replication(self):
        gpt_4_base_params = {
            "model": "gpt-4-base",
            "temperature": 0,
            "max_tokens": 50,
            "top_p": 1,
            "best_of": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "logprobs": None,
            "stop": None,
        }

        gpt_4_base = GPT(**gpt_4_base_params)
        prompt = "Who would win, an elephant or an ant?"

        response, _, _ = gpt_4_base.call(prompt=prompt)

        expected_start = " Well, an ant would win"

        self.assertTrue(response.startswith(expected_start))

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
        self.chat_model_test(model="gpt-4-32k-0613")

    def test_chat_model_fast(self):
        self.chat_model_test(model="gpt-4o-mini")
