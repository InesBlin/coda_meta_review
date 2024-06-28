# -*- coding: utf-8 -*-
"""
Prep data for automated hypothesis generation

- LLM: `data` to feed (ie tuples of values that form a hypothesis)
"""
import os
import math
import click
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

def type_of_effect(row):
    """ Categorize effect based on its signifiance """
    if math.isnan(row.ESLower) or math.isnan(row.ESUpper):
        if row.ES > -0.2 and row.ES < 0.2:
            return 'noEffect'
        return 'positive' if row.ES >= 0.2 else 'negative'
    if row.ESLower <= 0 <= row.ESUpper:
        return 'noEffect'
    return 'positive'  if float(row.ES) > 0 else 'negative'


def prep_data_llm(data: pd.DataFrame, th: str):
    """ Prep data for LLM prompting """
    columns = {
        "regular": ["dependent", "giv_prop", "iv", "iv_label", "cat_t1", "cat_t1_label", "cat_t2", "cat_t2_label", ],
        "var_mod": ["dependent", "giv_prop","iv", "iv_label", "cat_t1", "cat_t1_label", "cat_t2", "cat_t2_label", "mod", "mod_label", "mod_t1", "mod_t1_label", "mod_t2", "mod_t2_label", ],
        "study_mod": ["dependent", "giv_prop","iv", "iv_label", "cat_t1", "cat_t1_label", "cat_t2", "cat_t2_label", "mod", "mod_label", "mod_val", "mod_val_label", ]
    }
    return data.groupby(columns[th]).agg({"obs": "count"}).reset_index()


def prep_data_classification(data: pd.DataFrame, th: str):
    """ Prep data for classification (mostly: adding target to predict)"""
    columns = {
        "regular": ["giv_prop", "iv", "iv_label", "cat_t1", "cat_t1_label", "iv", "iv_label", "cat_t2", "cat_t2_label"],
        "var_mod": ["giv_prop", "iv",  "iv_label", "cat_t1", "cat_t1_label", "iv", "iv_label", "cat_t2", "cat_t2_label", "mod", "mod_label", "mod_t1", "mod_t1_label", "mod_t2", "mod_t2_label"],
        "study_mod": ["giv_prop", "iv", "iv_label", "cat_t1", "cat_t1_label", "iv", "iv_label", "cat_t2", "cat_t2_label", "mod", "mod_label", "mod_val", "mod_val_label"]
    }
    tqdm.pandas()
    data["effect"] = data.progress_apply(type_of_effect, axis=1)
    # Only keeping data with positive/negative effect
    data = data[data.effect != "noEffect"]
    cols = columns[th] + ["dependent", "effect"]

    for col in columns[th]:
        if not col.endswith("_label"):
            data = data[data[col].str.startswith("http")]
    
    return data[cols]


def prep_data_lp(data: pd.DataFrame, th:str):
    """ Prep data LP """
    counting = defaultdict(int)
    iv_new_unique = []
    for val in tqdm(data.iv_new.values):
        counting[val] += 1
        iv_new_unique.append(f"{val}_{str(counting[val])}")
    data["iv_new_unique"] = iv_new_unique
    return data

def prep_data(data: pd.DataFrame, type_method: str, th: str):
    """ Preparing data to be used for automated hypothesis generation """
    if type_method == "llm":
        return prep_data_llm(data=data, th=th)
    
    if type_method == "classification":
        return prep_data_classification(data=data, th=th)
    
    if type_method == "lp":
        return prep_data_lp(data=data, th=th)


@click.command()
@click.argument("folder_in")
@click.argument("folder_out")
@click.argument("type_method")
def main(folder_in, folder_out, type_method):
    """"""
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    files = os.listdir(folder_in)
    for file in tqdm(files):
        save_path = os.path.join(folder_out, file)
        if not os.path.exists(save_path):
            th = file.replace("h_", "").split("_es")[0]
            data = pd.read_csv(os.path.join(folder_in, file), index_col=0)
            output = prep_data(data=data, th=th, type_method=type_method)
            output.to_csv(save_path)



if __name__ == '__main__':
    # python experiments/prep_data.py data/hypotheses/entry/ data/hypotheses/llm llm
    # python experiments/prep_data.py data/hypotheses/entry/ data/hypotheses/classification classification
    # python experiments/prep_data.py data/hypotheses/entry/ data/hypotheses/lp lp
    main()
