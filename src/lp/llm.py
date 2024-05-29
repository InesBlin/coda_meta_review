# -*- coding: utf-8 -*-
"""
LLM Method for hypothesis generation
"""
from openai import OpenAI
from src.settings import API_KEY_GPT

CLIENT = OpenAI(api_key=API_KEY_GPT)
MODEL = "gpt-3.5-turbo-0125"


def run_gpt(prompt: str, content: Union[str, List[str]], **add_content):
    """ Get answer from GPT from prompt + content """
    if isinstance(content, str):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content}
        ]
    else:  # list of text
        messages = \
            [{"role": "system", "content": prompt}] + \
            [{"role": "user", "content": c} for c in content]

    if add_content and add_content.get("entities"):
        messages += [{"role": "user", "content": add_content.get("entities")}]
    completion = CLIENT.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0)
    return completion.choices[0].message.content

class LLMHypothesisGeneration:
    """ LLM-based hypothesis generation """
    def __init__(self, type_hypothesis):
        self.prompt_template = """
        You will be given data in .csv format, where each line can be read as a hypothesis on how humans cooperate. The csv has the following columns: [columns].

        Each line in the data can be templated as a hypothesis as follows:
        "[templated_h]".
        [cols_data] comes from the data.


        Given the data, you must extract the subset of 5 rows that represent the most coherent hypotheses. [cols_add] 

        The output must be in csv format.

        The output is:
        ```csv
        ```
        """

        self.custom_content = {
            "regular": {
                "[columns]": "`giv_prop`, `iv`, `cat_t1` and `cat_t2`",
                "[templated_h]": "Cooperation is significantly {higher/lower} when `{iv}` is `{cat_t1}` compared to when `{iv}` is `{cat_t2}`.",
                "[cols_data]": "`iv`, `cat_t1` and `cat_t2`",
                "[cols_add]": "On top of the existing columns, you must add another one, `comparative` that can take the values `higher` or `lower`."
            }
        }