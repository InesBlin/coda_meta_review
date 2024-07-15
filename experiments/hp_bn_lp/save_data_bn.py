# -*- coding: utf-8 -*-
"""
Save data for blank node hypotheses over KG
"""
import os
import math
import click
import pandas as pd
from tqdm import tqdm
from loguru import logger
from src.lp.build_blank_h_kg import BlankHypothesesKGBuilder

TYPE_HYPOTHESIS = ['regular', 'var_mod', 'study_mod']
ES_MEASURE = ['d']
BHKGB = BlankHypothesesKGBuilder()

def type_of_effect(row):
    """ Categorize effect based on its signifiance """
    if math.isnan(row.ESLower) or math.isnan(row.ESUpper):
        if row.ES > -0.2 and row.ES < 0.2:
            return 'noEffect'
        return 'positive' if row.ES >= 0.2 else 'negative'
    if row.ESLower <= 0 <= row.ESUpper:
        return 'noEffect'
    return 'positive'  if float(row.ES) > 0 else 'negative'

def get_data(folder_in, th, esm):
    """ Get data with positive/negative effect only """
    data = pd.read_csv(os.path.join(folder_in, f"h_{th}_es_{esm}.csv"), index_col=0)
    tqdm.pandas()
    data["effect"] = data.progress_apply(type_of_effect, axis=1)
    return data[data.effect != "noEffect"]

@click.command()
@click.argument("folder_in")
@click.argument("folder_out")
@click.argument("vocab")
def main(folder_in, folder_out, vocab):
    """ Build KG for LP task for each file """
    vocab = pd.read_csv(vocab).fillna("")
    vocab.columns = ["subject", "predicate", "object"]
    for col in vocab.columns:
        vocab = vocab[vocab[col].str.startswith("http")]

    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    for th in TYPE_HYPOTHESIS:
        for esm in ES_MEASURE: 
            logger.info(f"Building KG for hypothesis `{th}` with effect size measure `{esm}`")
            save_path_with = os.path.join(folder_out, f"h_{th}_es_{esm}")
            if not os.path.exists(f"{save_path_with}_random.csv"):
                data = get_data(folder_in, th, esm)
                output_random, output_effect = BHKGB(data=data, vocab=vocab)
                output_random.to_csv(f"{save_path_with}_random.csv")
                output_effect.to_csv(f"{save_path_with}_effect.csv")


if __name__ == '__main__':
    # python experiments/hp_bn_lp/save_data_bn.py data/hypotheses/lp/ data/hypotheses/bn/ data/vocab.csv
    main()