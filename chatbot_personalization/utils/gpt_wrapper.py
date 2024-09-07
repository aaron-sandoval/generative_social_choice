import openai
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_not_exception_type,
)
import os
from chatbot_personalization.utils.helper_functions import (
    get_base_dir_path,
    get_time_string,
)
import pandas as pd
import numpy as np
from typing import Any, Optional, Tuple
from pydantic import BaseModel


OPENAI_API_KEY_PATH = get_base_dir_path() / "utils" / "OPENAI_API_KEY"
OPENAI_ORGANIZATION_PATH = get_base_dir_path() / "utils" / "OPENAI_ORGANIZATION"

MAX_QUERY_RETRIES = 5

Message = dict[str, Any]
MessageList = list[Message]


class LLMLog(BaseModel):
    # These fields are essential, but some LLMLog objects also have more fields
    model: str
    params: dict[str, Any]
    prompt: str
    system_prompt: str
    messages: MessageList
    response: str
    completion: Any
    timestamp: str
    system_fingerprint: str


def get_probabilities_from_completion(completion, token_idx):
    """
    Given a GPT completion object, and token_idx of the token we care about, return the actual probabilitiy distribution of that token.
    """
    completion = completion.choices
    assert len(completion) == 1
    try:
        logprobs = completion[0].logprobs.content[token_idx].top_logprobs
        logprobs = pd.Series({entry.token: entry.logprob for entry in logprobs})
    except AttributeError:
        # TODO: instead of catching these, we should check the model type and use the appropriate method
        logprobs = completion[0].logprobs.top_logprobs[token_idx]
        logprobs = pd.Series(logprobs)
    except IndexError:
        return pd.Series([])
    probs = logprobs.apply(np.exp)
    return probs


def get_token_idx_of_fields_value(
    completion: Any, logprobs_field: str
) -> Optional[int]:
    """
    Given a completion object (assumed to be JSON), and a field name (assumed to be contained in that JSON), return the index of the token corresponding to the value written after that field name (asumed to be a single-token repsonse).

    For example, if output JSON contains {... some stuff..., "score" : 2}, and logprobs_field='score', then this identifies the index of the token containing '2'.

    If this fails, return None.
    """
    # Figure out which token corresponds to logprobs_field
    # To do this: keep chopping off tokens at the end, until logprobs_field is no longer contained in it
    assert logprobs_field in completion.choices[0].message.content
    for field_idx in range(len(completion.choices[0].logprobs.content), 0, -1):
        initial_message = "".join(
            [
                token_logprob.token
                for token_logprob in completion.choices[0].logprobs.content[:field_idx]
            ]
        )

        if logprobs_field not in initial_message:
            break
    else:
        raise KeyError(f"Could not find key {logprobs_field}")
    # field_idx is the token idx of the last token of logprobs_field

    # Figure out which token corresponds to the value of logprobs_field (should be shortly after)
    for idx, token_logprob in enumerate(
        completion.choices[0].logprobs.content[field_idx + 1 :]
    ):
        token = token_logprob.token
        try:
            _ = int(token)
            return idx + field_idx + 1
        except ValueError:
            continue  #  Haven't found token with score yet
    return None


def is_completion_model(model: str) -> bool:
    if model in [
        "gpt-4-base",
        "davinci",
        "davinci-002",
        "text-davinci-003",
        "gpt-3.5-turbo-instruct",
    ]:
        return True
    else:
        return False


class GPT:
    def __init__(self, *, model, openai_seed=0, **params):
        ### Get OpenAI API key and organization

        if OPENAI_API_KEY_PATH.exists():
            with open(OPENAI_API_KEY_PATH, "r") as f:
                api_key = f.readline().strip()
        else:
            api_key = os.getenv("OPENAI_API_KEY")

        if OPENAI_ORGANIZATION_PATH.exists():
            with open(OPENAI_ORGANIZATION_PATH, "r") as f:
                organization = f.readline().strip()
        else:
            organization = None

        self.client = openai.OpenAI(
            api_key=api_key,
            organization=organization,
        )
        self.model = model
        self.params = params
        self.openai_seed = openai_seed

    @retry(
        stop=stop_after_attempt(MAX_QUERY_RETRIES),
        wait=wait_random_exponential(),
        retry=retry_if_not_exception_type(
            (openai.BadRequestError, AssertionError)
        ),  # this means too many tokens
    )
    def _call_completion(self, prompt: str, **params) -> Tuple[str, Any]:
        """
        Make OpenAI API call to a legacy completions style model (e.g. gpt-4-base)

        Return: response (str), completion (openai.Completion object)
        """
        assert is_completion_model(self.model)
        completion = self.client.completions.create(
            model=self.model, prompt=prompt, seed=self.openai_seed, **params
        )
        response = completion.choices[0].text
        system_fingerprint = completion.system_fingerprint
        return response, completion, system_fingerprint

    @retry(
        stop=stop_after_attempt(MAX_QUERY_RETRIES),
        wait=wait_random_exponential(),
        retry=retry_if_not_exception_type(
            (openai.BadRequestError, AssertionError)
        ),  # this means too many tokens
    )
    def _call_chat(self, messages: MessageList, **params) -> Tuple[str, Any]:
        """
        Make OpenAI API call to a chat style model (e.g. gpt-4-0314, gpt-4-turbo, gpt-4o)

        Return: response (str), completion (openai.Completion object)
        """
        assert not is_completion_model(self.model)
        completion = self.client.chat.completions.create(
            model=self.model, messages=messages, seed=self.openai_seed, **params
        )
        response = completion.choices[0].message.content
        system_fingerprint = completion.system_fingerprint
        return response, completion, system_fingerprint

    def call(
        self,
        messages: Optional[MessageList] = None,
        prompt: Optional[str] = None,
        **local_params,
    ) -> Tuple[str, Any, LLMLog]:
        """
        Make LLM call. Specify EITHER messages object OR prompt:
        - If using legacy completion model (e.g. gpt-4-base), must use prompt.
        - Otherwise (e.g. gpt-4o), passing messages object is recommended, but prompt also works.
        """
        if "messages" in local_params or "messages" in self.params:
            raise DeprecationWarning(
                "Having GPT object store messages is deprecated. Instead pass messages object or prompt. An example messages object is messages=[{'role': 'system', 'content': 'your system prompt'}, {'role': 'user', 'content': 'your main prompt'}]."
            )
        params = self.params.copy()
        params.update(local_params)

        if messages is None and prompt is None:
            raise ValueError("Either messages or prompt must be provided.")
        elif messages is not None and prompt is not None:
            raise ValueError("Can't provide both messages and prompt")
        elif messages is None and prompt is not None:
            # prompt provided
            if is_completion_model(self.model):
                response, completion, system_fingerprint = self._call_completion(
                    prompt=prompt, **params
                )
            else:
                response, completion, system_fingerprint = self._call_chat(
                    messages=[{"role": "user", "content": prompt}], **params
                )
            # For logging specify that there was no system prompt
            system_prompt = None
        elif messages is not None and prompt is None:
            # messages provided
            response, completion, system_fingerprint = self._call_chat(
                messages=messages, **params
            )
            # For logging, extract prompt and system prompt
            system_prompt, prompt = "", ""
            for message in messages:
                if message["role"] == "system":
                    system_prompt += message["content"]
                elif message["role"] == "user":
                    prompt += message["content"]
        else:
            raise ValueError("????")

        log = {
            "model": self.model,
            "openai_seed": self.openai_seed,
            "params": params,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "messages": messages,
            "response": response,
            "completion": completion,
            "system_fingerprint": system_fingerprint,
            "timestamp": get_time_string(),
        }

        return response, completion, log
