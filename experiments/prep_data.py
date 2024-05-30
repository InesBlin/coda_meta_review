# -*- coding: utf-8 -*-
"""
Prep data for automated hypothesis generation

- LLM: `data` to feed (ie tuples of values that form a hypothesis)
"""
import os
import click
from tqdm import tqdm
import pandas as pd

def prep_data_llm(data: pd.DataFrame, type_data: str):
    """ Prep data for LLM prompting """
    columns = {
        "regular": ["giv_prop", "iv", "iv_label", "cat_t1", "cat_t1_label", "cat_t2", "cat_t2_label", ],
        "var_mod": ["giv_prop","iv", "iv_label", "cat_t1", "cat_t1_label", "cat_t2", "cat_t2_label", "mod", "mod_label", "mod_t1", "mod_t1_label", "mod_t2", "mod_t2_label", ],
        "study_mod": ["giv_prop","iv", "iv_label", "cat_t1", "cat_t1_label", "cat_t2", "cat_t2_label", "mod", "mod_label", "mod_val", "mod_val_label", ]
    }
    return data.groupby(columns[type_data]).agg({"obs": "count"}).reset_index()

def prep_data(data: pd.DataFrame, type_method: str, type_data: str):
    """ Preparing data to be used for automated hypothesis generation """
    if type_method == "llm":
        return prep_data_llm(data=data, type_data=type_data)


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
            type_data = file.replace("h_", "").split("_es")[0]
            data = pd.read_csv(os.path.join(folder_in, file), index_col=0)
            output = prep_data(data=data, type_method=type_method,
                               type_data=type_data)
            output.to_csv(save_path)


if __name__ == '__main__':
    # python experiments/prep_data.py data/hypotheses/entry/ data/hypotheses/llm llm
    main()
