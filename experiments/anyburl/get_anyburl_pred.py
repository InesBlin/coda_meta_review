# -*- coding: utf-8 -*-
"""
Readable outputs from AnyBURL
"""
import os
import click
import pandas as pd
from src.knowledge import generate_hypothesis

TYPE_HYPOTHESIS = ['regular', 'var_mod', 'study_mod']
ES_MEASURE = ['d']

VALS = ['giv_prop', 'iv', 'cat_t1', 'cat_t2', 'mod', 'mod_t1', 'mod_t2', 'mod_val']
COLUMNS = ['dependent', 'iv_new_unique'] + VALS + [f"{x}_label" for x in VALS]
BN_COL = 'iv_new_unique'

COMPARATIVES = {
    "https://data.cooperationdatabank.org/vocab/prop/hasNegativeEffectOn": "lower",
    "https://data.cooperationdatabank.org/vocab/prop/hasPositiveEffectOn": "higher"
}

def get_hypotheses(pred_path):
    """ Get predictions from output of AnyBURL """
    not_start = ["Heads", "Tails"]
    hypotheses = open(pred_path, 'r', encoding='utf-8').readlines()
    hypotheses = [x.replace("\n", "").split(' ') for x in hypotheses if not any(x.startswith(y) for y in not_start)]
    return {s: p for s, p, _ in hypotheses}

def get_all_data(data_path, pred_path, th):
    """ Putting everything together, by combining:
    (1) original data with information about hypotheses
    (2) predictions """
    df = pd.read_csv(data_path, index_col=0)
    df = df[[c for c in df.columns if c in COLUMNS]]
    hypotheses = get_hypotheses(pred_path=pred_path)
    df = df[df.iv_new_unique.isin(hypotheses.keys())]
    df["comparative"] = df.apply(lambda row: COMPARATIVES[hypotheses[row["iv_new_unique"]]], axis=1)
    df = df.apply(lambda row: generate_hypothesis(row, th), axis=1)
    return df[[c for c in df.columns if c != BN_COL]]

def save_top_hypothesis(df, folder_save, top_n: int = 5):
    """ Saving top scored hypotheses for each giv in the test data
    Assuming that hypotheses are already ordered by descending confidence. """
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
    for giv in df.giv_prop.unique():
        name = giv.split("/")[-1]
        curr_df = df[df.giv_prop == giv]
        top = min(curr_df.shape[0], top_n)
        curr_df.to_csv(os.path.join(folder_save, f"{name}.csv"))
        f = open(os.path.join(folder_save, f"{name}.txt"), 'w', encoding='utf-8')
        for _, row in curr_df[:top].iterrows():
            f.write(f"{row.hypothesis}\n")
        f.close()

@click.command()
@click.argument("folder_in_data")
@click.argument("folder_in_preds")
@click.argument("folder_out")
def main(folder_in_data, folder_in_preds, folder_out):
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    for th in TYPE_HYPOTHESIS:
        for es in ES_MEASURE:
            save_f = os.path.join(folder_out, f"h_{th}_es_{es}")
            if not os.path.exists(save_f):
                os.makedirs(save_f)
            if not os.path.exists(os.path.join(save_f, "data.csv")):
                data = get_all_data(
                    data_path=os.path.join(folder_in_data, f"h_{th}_es_{es}.csv"),
                    pred_path=os.path.join(folder_in_preds, f"h_{th}_es_{es}"),
                    th=th)
                data.to_csv(os.path.join(save_f, "data.csv"))
                save_top_hypothesis(df=data, folder_save=os.path.join(save_f, "outputs"), top_n=5)


if __name__ == '__main__':
    # python experiments/anyburl/get_anyburl_pred.py data/hypotheses/lp experiments/anyburl/preds experiments/anyburl/final 
    main()
