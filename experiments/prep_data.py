# -*- coding: utf-8 -*-
"""
Prep data for automated hypothesis generation

- LLM: `data` to feed (ie tuples of values that form a hypothesis)
"""
import pandas as pd

def prep_data_llm(data: pd.DataFrame, type_data: str):
    """ Prep data for LLM prompting """
    columns = {
        "regular": ["giv_prop","iv", "cat_t1", "cat_t2"],
        "var_mod": ["giv_prop","iv", "cat_t1", "cat_t2", "mod", "mod_t1", "mod_t2"],
        "study_mod": ["giv_prop","iv", "cat_t1", "cat_t2", "mod", "mod_val"]
    }
    return data.groupby(columns[type_data]).agg({"obs": "count"}).reset_index()

def prep_data(data: pd.DataFrame, type_method: str, type_data: str):
    """ Preparing data to be used for automated hypothesis generation """
    if type_method == "llm":
        return prep_data_llm(data=data, type_data=data)
