# -*- coding: utf-8 -*-
"""
Prep data for automated hypothesis generation

- LLM: `data` to feed (ie tuples of values that form a hypothesis)
"""
import os
import math
import click
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
        "regular": ["giv_prop", "iv", "iv_label", "cat_t1", "cat_t1_label", "cat_t2", "cat_t2_label", ],
        "var_mod": ["giv_prop","iv", "iv_label", "cat_t1", "cat_t1_label", "cat_t2", "cat_t2_label", "mod", "mod_label", "mod_t1", "mod_t1_label", "mod_t2", "mod_t2_label", ],
        "study_mod": ["giv_prop","iv", "iv_label", "cat_t1", "cat_t1_label", "cat_t2", "cat_t2_label", "mod", "mod_label", "mod_val", "mod_val_label", ]
    }
    return data.groupby(columns[th]).agg({"obs": "count"}).reset_index()


def prep_data_classification(data: pd.DataFrame, th: str):
    """ Prep data for classification (mostly: adding target to predict)"""
    columns = {
        "regular": ["giv_prop", "iv", "cat_t1", "iv", "cat_t2"],
        "var_mod": ["giv_prop", "iv", "cat_t1", "iv", "cat_t2", "mod", "mod_t1", "mod_t2"],
        "study_mod": ["giv_prop", "iv", "cat_t1", "iv", "cat_t2", "mod", "mod_val"]
    }
    tqdm.pandas()
    data["effect"] = data.progress_apply(type_of_effect, axis=1)
    cols = columns[th] + ["dependent", "effect"]

    for col in columns[th]:
        data = data[data[col].str.startswith("http")]
    
    return data[cols]

def prep_data(data: pd.DataFrame, type_method: str, th: str):
    """ Preparing data to be used for automated hypothesis generation """
    if type_method == "llm":
        return prep_data_llm(data=data, th=th)
    
    if type_method == "classification":
        return prep_data_classification(data=data, th=th)


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
    main()
