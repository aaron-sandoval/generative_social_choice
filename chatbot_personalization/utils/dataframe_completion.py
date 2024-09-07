# %%
from io import StringIO
import textwrap
import pandas as pd

import chatbot_personalization.utils.gpt_wrapper as gpt_wrapper


def wprint(text, width=80):
    if isinstance(text, list):
        text = "[\n" + "\n\n------------\n\n".join(text) + "\n]"

    lines = text.split("\n")
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    wrapped_text = "\n".join(wrapped_lines)
    print(wrapped_text)


class DataFrameCompleter:
    def __init__(
        self,
        *,
        llm: gpt_wrapper.GPT,
        todo_marker: str = "LLM_TODO",
        done_marker: str = "LLM_DONE\\n",
    ) -> None:
        self.llm = llm
        self.todo_marker = todo_marker
        self.done_marker = done_marker

    def complete(
        self, df: pd.DataFrame, system_prompt: str, verbose: bool = False
    ) -> pd.DataFrame:
        df_in = df

        df = df.to_json(orient="records")
        df = df.split(self.todo_marker, maxsplit=1)
        assert len(df) == 2

        stop = df[1][:2]

        prompt = df[0].replace(self.done_marker, "")
        assert "LLM_" not in prompt

        response, completion, log = self.llm.call(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stop=stop,
        )

        # json does not like quotes without backslash
        response = response.replace('\\"', '"').replace('"', '\\"')
        df_complete = "".join([df[0], self.done_marker + response, df[1]])

        if verbose:
            print("stop: ", stop)

            print("\n\n******* df prefix: ********************************\n")
            wprint(df[0][-500:])
            print("\n*****************************************************")
            print("\n\n******* response: *********************************\n")
            wprint(response)
            print("\n*****************************************************")
            print("\n\n******* df suffix: ********************************\n")
            wprint(df[1][:200])
            print("\n*****************************************************")
            print("\n\n******* full prompt: ******************************\n")
            wprint(prompt)
            print("\n*****************************************************")

        assert completion.choices[0].finish_reason == "stop"
        df_complete = pd.read_json(StringIO(df_complete))
        # df_complete = pd.read_json(df_complete)
        df_complete.index = df_in.index
        assert (df_in != df_complete).sum().sum() == 1
        return df_complete, log
