# -*- coding: utf-8 -*-
"""
Saving data in correct format for AnyBURL LP
"""
import os
import click
from loguru import logger
from typing import Tuple
import pandas as pd
from pykeen.triples import TriplesFactory
from src.lp.blank_node_lp import check_tvt_prop, split_effect_triples

TYPE_HYPOTHESIS = ['regular', 'var_mod', 'study_mod']
ES_MEASURE = ['d']
TVT_SPLIT = [0.8, 0.1, 0.1]

def split(data_reg: pd.DataFrame, data_effect: pd.DataFrame,
          tvt_split: Tuple[float, float, float], th: str):
    check_tvt_prop(tvt_split=tvt_split)
    spo_cols = ["subject", "predicate", "object"]
    sh = TriplesFactory.from_labeled_triples(data_reg[spo_cols].values)
    c_split = split_effect_triples(data=data_effect, tvt_split=tvt_split, th=th)

    res = {
        "train": pd.concat([pd.DataFrame(sh.triples, columns=spo_cols), c_split["train"]]),
        "val": c_split["val"],
        "test": c_split["test"],
    }
    return res

@click.command()
@click.argument("folder_in")
@click.argument("folder_out")
def main(folder_in, folder_out):
    """ Main to save train/val/test data """
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    for th in TYPE_HYPOTHESIS:
        for esm in ES_MEASURE:
            logger.info(f"Saving data for AnyBURL for `{th}` with effect size measure `{esm}`")
            save_path = os.path.join(folder_out, f"train_h_{th}_es_{esm}.csv")
            if not os.path.exists(save_path):
                data_reg = pd.read_csv(
                    os.path.join(folder_in, f"h_{th}_es_{esm}_random.csv"), index_col=0).dropna()
                data_effect = pd.read_csv(
                    os.path.join(folder_in, f"h_{th}_es_{esm}_effect.csv"), index_col=0).dropna()
                res = split(data_reg, data_effect, TVT_SPLIT, th)
                for key, df in res.items():
                    df.to_csv(save_path.replace("train_", f"{key}_"), sep="\t", index=False, header=False)


if __name__ == '__main__':
    # python experiments/save_data_anyburl.py ./data/hypotheses/bn/with_vocab ./data/hypotheses/anyburl/with_vocab
    # python experiments/save_data_anyburl.py ./data/hypotheses/bn/without_vocab ./data/hypotheses/anyburl/without_vocab
    main()