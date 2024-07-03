# -*- coding: utf-8 -*-
"""
LLM Method for hypothesis generation
"""
from io import StringIO
import pandas as pd
from openai import OpenAI
from src.settings import API_KEY_GPT

CLIENT = OpenAI(api_key=API_KEY_GPT)
MODEL = "gpt-3.5-turbo-0125"


def run_gpt(prompt: str, content: str, model: str):
    """ Get answer from GPT from prompt + content """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content}
    ]
    completion = CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0)
    return completion.choices[0].message.content

class LLMHypothesisGeneration:
    """ LLM-based hypothesis generation """
    def __init__(self, type_hypothesis, model: str = MODEL):
        self.model = model
        self.th = type_hypothesis
        self.prompt_template = """
        You will be given data in .csv format, where each line can be read as a hypothesis on how humans cooperate. The csv has the following columns: [columns].

        Each line in the data can be templated as a hypothesis as follows:
        "[templated_h]".
        [cols_data] comes from the data ; [new_cols_data] must be added and can only take the values `higher` or `lower`.


        Given the data, you must extract the subset of 5 rows that represent the most coherent hypotheses, ranked from the most coherent one. If there are less than 5 rows, you can re-order them by order of coherence. [cols_add] 

        The output must be in csv format.

        The output is:
        ```csv
        ```
        """
        self.cols = {
            "regular": ['dependent', 'iv_label', 'cat_t1_label', 'cat_t2_label', 'comparative'],
            "var_mod": ['dependent', 'iv_label', 'cat_t1_label', 'cat_t2_label', 'mod_label', 'mod_t1_label', 'mod_t2_label', 'comparative'],
            "study_mod": ['dependent', 'iv_label', 'cat_t1_label', 'cat_t2_label', 'mod_label', 'mod_val_label', 'comparative']
        }

        self.custom_content = {
            "regular": {
                "[columns]": "`dependent`, `giv_prop`, `iv`, `iv_label`, `cat_t1`, `cat_t1_label`, `cat_t2` and `cat_t2_label`.",
                "[templated_h]": "Cooperation  is significantly `{comparative}` when `{iv_label}` is `{cat_t1_label}` compared to when `{iv_label}` is `{cat_t2_label}`.",
                "[cols_data]": "`iv_label`, `cat_t1_label` and `cat_t2_label`",
                "[new_cols_data]": "`comparative`",
                "[cols_add]": "On top of the existing columns, you must add another one, `comparative` that can take the values `higher` or `lower`."
            },
            "var_mod": {
                "[columns]": "`dependent`, `giv_prop`, `iv`, `iv_label`, `cat_t1`, `cat_t1_label`, `cat_t2`, `cat_t2_label`, `mod`, `mod_label`, `mod_t1`, `mod_t1_label`, `mod_t2`  and `mod_t2_label`.",
                "[templated_h]": "When comparing studies where `{iv_label}` is `{cat_t1_label}` and studies where `{iv_label}` is `{cat_t2_label}`, cooperation from studies involving `{mod_t1_label}` as `{mod_label}` is significantly `{comparative}` than cooperation from studies involving based on `{mod_t2_label}` as `{mod_label}`.",
                "[cols_data]": "`iv_label`, `cat_t1_label`, `cat_t2_label`, `mod_label`, `mod_t1_label`, and `mod_t2_label`",
                "[new_cols_data]": "`comparative`",
                "[cols_add]": "On top of the existing columns, you must add another one, `comparative` that can take the values `higher` or `lower`."
            },
            "study_mod": {
                "[columns]": "`dependent`, `giv_prop`, `iv`, `iv_label`, `cat_t1`, `cat_t1_label`, `cat_t2`, `cat_t2_label`, `mod`, `mod_label`, `mod_val` and `mod_val_label`.",
                "[templated_h]": "When comparing studies where `{iv_label}` is `{cat_t1_label}` and studies where `{iv_label}` is `{cat_t2_label}`, cooperation is significantly `{comparative}` when `{mod_label}` is `{mod_val_label}` compared to when `{mod_label}` has another value.",
                "[cols_data]": "`iv_label`, `cat_t1_label`, `cat_t2_label`, `mod_label`, and `mod_val_label`",
                "[new_cols_data]": "`comparative`",
                "[cols_add]": "On top of the existing columns, you must add another one, `comparative` that can take the values `higher` or `lower`."
            }
        }

        if type_hypothesis not in self.custom_content.keys():
            raise ValueError(f"This type of hypothesis is not supported. Current supported are: {self.custom_content.keys()}")

        self.prompt = self.prompt_template
        for key, content in self.custom_content[type_hypothesis].items():
            self.prompt = self.prompt.replace(key, content)
    
    def generate_readable_h(self, row):
        """ Generate readable hypothesis from a pandas row """
        hypothesis = self.custom_content[self.th]["[templated_h]"]
        for col in self.cols[self.th]:
            hypothesis = hypothesis.replace("`{"+str(col)+"}`", str(row[col]))
        return hypothesis
    
    def __call__(self, data):
        """ Prompt with data """
        return run_gpt(prompt=self.prompt, content=data, model=self.model)


if __name__ == '__main__':
    for th in ['regular', 'var_mod', 'study_mod']:
        LLMHG = LLMHypothesisGeneration(type_hypothesis=th)
        print(f"TH: {th}\n{LLMHG.prompt}\n=====\n")
