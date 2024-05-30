# -*- coding: utf-8 -*-
"""
Save data for blank node hypotheses over KG
"""
import os
import click
import pandas as pd
from loguru import logger
from src.lp.build_blank_h_kg import BlankHypothesesKGBuilder

TYPE_HYPOTHESIS = ['regular', 'var_mod', 'study_mod']
ES_MEASURE = ['d', 'r']
BHKGB = BlankHypothesesKGBuilder()

@click.command()
@click.argument("folder_in")
@click.argument("folder_out")
@click.argument("vocab")
def main(folder_in, folder_out, vocab):
    """ Build KG for LP task for each file """
    vocab = pd.read_csv(vocab, index_col=0)
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    for th in TYPE_HYPOTHESIS:
        for esm in ES_MEASURE: 
            logger.info(f"Building KG for hypothesis `{th}` with effect size measure `{esm}`")
            save_path = os.path.join(folder_out, f"h_{th}_es_{esm}")
            if not os.path.exists(f"{save_path}_random.csv"):
                data = pd.read_csv(os.path.join(folder_in, f"h_{th}_es_{esm}.csv"), index_col=0)
                output_random, output_effect = BHKGB(data=data, vocab=vocab)
                output_random.to_csv(f"{save_path}_random.csv")
                output_effect.to_csv(f"{save_path}_effect.csv")


if __name__ == '__main__':
    # python experiments/save_data_bn.py data/hypotheses/entry/ data/hypotheses/bn/ data/vocab.csv
    main()