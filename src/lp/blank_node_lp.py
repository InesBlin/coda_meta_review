# -*- coding: utf-8 -*-
"""
Link Prediction for Blank Node
"""
import os
import torch
import random
from typing import Union, Tuple
import pandas as pd
from tqdm import tqdm
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.sampling import BasicNegativeSampler

torch.cuda.empty_cache()

def check_tvt_prop(tvt_split: Tuple[int, int, int]):
    """ Check that coherent, eg. sum = 1 """
    if sum(tvt_split) != 1:
        raise ValueError("The sum of the values in `train_val_test_split` must be 1.")
    if any(x > tvt_split[0] for x in tvt_split[1:]):
        raise ValueError("The split for `training` must be the highest.")


def init_empty_df():
    return pd.DataFrame(columns=["subject", "predicate", "object"])


def split_subject(subject: str, th: str):
    """ Split subject into a more readable format """
    if th == "regular":
        return subject.split("_")[0].split('/')[-1]
    # th == "study_mod" or th == 'var_mod'
    return '_'.join([y.split('/')[-1] for y in subject.split('_')[:2]])
    

def distribute(indexes, train, val, test):
    """ Extract `train`, `val`and `test` elements from the list of `indexes``
    such that their sum equals the indexes
    """
    random.seed(23)
    train_l = random.sample(indexes, train)
    remaining = list(set(indexes).difference(set(train_l)))

    random.seed(23)
    val_l = random.sample(remaining, val)
    test_l = list(set(remaining).difference(set(val_l)))
    
    return train_l, val_l, test_l


def split_effect_triples(data: pd.DataFrame, tvt_split: Tuple[int, int, int], th: str):
    """ Should distribute `effect data` across train/val/test split
    - `effect data`: only concerns the following predicates
        * cp:hasPositiveEffectOn
        * cp:hasNegativeEffectOn
        * cp:hasNoEffectOn
    - distribution: for each pairs of givs and predicate, proportional distribution

    Rule for distribution of a pair (h_id, pred), that appear `nb` time, with perc `prop`
    - If nb in [1, 2] -> random 
    - If nb = 3 -> 1 in each
    - If nb >= 4
        - If prop < 1/nb -> 1
        - Else: int(prop*nb)
    """
    check_tvt_prop(tvt_split=tvt_split)
    data["h_id"] = data["subject"].apply(lambda x: split_subject(x, th))
    data_no_reps = data.drop_duplicates()
    grouped = data_no_reps.groupby(["h_id", "predicate"]).agg({"subject": "count"}).reset_index().rename(columns={"subject": "nb"})

    cols_triples = ["subject", "predicate", "object"]
    col_td = ["train", "val", "test"]
    res = {x: init_empty_df() for x in col_td}

    for _, row in tqdm(grouped.iterrows(), total=grouped.shape[0]):
        curr_data = data_no_reps[(data_no_reps.predicate==row.predicate) & (data_no_reps.h_id==row.h_id)]
        curr_data = curr_data[cols_triples].reset_index()
        if row.nb < 3:
            random.seed(23)
            shuffled_td = random.sample(col_td, len(col_td))
            random.seed(23)
            sampled = random.sample(shuffled_td, row.nb)
            for index, row_curr_data in curr_data.iterrows():
                td = sampled[index]
                filter_data = (data[cols_triples[0]] == row_curr_data[cols_triples[0]]) & \
                                (data[cols_triples[1]] == row_curr_data[cols_triples[1]]) & \
                                    (data[cols_triples[2]] == row_curr_data[cols_triples[2]])
                res[td] = pd.concat([res[td], data[filter_data]])
        
        elif row.nb == 3:
            random.seed(23)
            sampled = random.sample(col_td, len(col_td))
            for index, row_curr_data in curr_data.iterrows():
                td = sampled[index]
                filter_data = (data[cols_triples[0]] == row_curr_data[cols_triples[0]]) & \
                                (data[cols_triples[1]] == row_curr_data[cols_triples[1]]) & \
                                    (data[cols_triples[2]] == row_curr_data[cols_triples[2]])
                res[td] = pd.concat([res[td], data[filter_data]])
        
        else:
            nb_test = max(int(tvt_split[2]*row.nb), 1)
            nb_val = max(int(tvt_split[1]*row.nb), 1)
            nb_train = row.nb - nb_test - nb_val
            train_i, val_i, test_i = distribute(list(curr_data.index), nb_train, nb_val, nb_test)
            for indexes, td in [(train_i, "train"), (val_i, "val"), (test_i, "test")]:
                for i in indexes:
                    row = curr_data.iloc[i]
                    filter_data = (data[cols_triples[0]] == row[cols_triples[0]]) & \
                                    (data[cols_triples[1]] == row[cols_triples[1]]) & \
                                        (data[cols_triples[2]] == row[cols_triples[2]])
                    res[td] = pd.concat([res[td], data[filter_data]])
            
    return res


def custom_split(data_reg: pd.DataFrame, data_effect: pd.DataFrame, tvt_split_reg: Tuple[int, int, int], tvt_split_effect: Tuple[int, int, int], th: str):
    check_tvt_prop(tvt_split=tvt_split_reg)
    check_tvt_prop(tvt_split=tvt_split_effect)
    spo_cols = ["subject", "predicate", "object"]
    sh = TriplesFactory.from_labeled_triples(data_reg[spo_cols].values)
    # sh_train, sh_val, sh_test = sh.split(tvt_split_reg, random_state=23)
    c_split = split_effect_triples(data=data_effect, tvt_split=tvt_split_effect, th=th)
    res = {
        "train": pd.concat([pd.DataFrame(sh.triples, columns=spo_cols), c_split["train"]]),
        "val": c_split["val"],
        "test": c_split["test"],
    }
    return {x: TriplesFactory.from_labeled_triples(val[spo_cols].values) for x, val in res.items()}


class BNLinkPredictor:
    """ LP for blank node hypotheses """
    def __init__(self, dr: str, de: str, th: str,
                 tvt_reg: Tuple[int, int, int] = [0.8, 0.1, 0.1],
                 tvt_effect: Tuple[int, int, int] = [0.8, 0.1, 0.1]):
        self.dr = pd.read_csv(dr, index_col=0).dropna()
        self.de = pd.read_csv(de, index_col=0).dropna()

        self.triples = custom_split(
            data_reg=self.dr, data_effect=self.de,
            tvt_split_reg=tvt_reg, tvt_split_effect=tvt_effect, th=th)
        print({k: v.num_triples for k, v in self.triples.items()})
    
    def init_hp_pipeline(self, model: str = "transe", random_seed: int = 23,
                      epochs: int = 250, embedding_dim: int = 256,
                      lr: float = 0.01, num_negs_per_pos: int = 50):
        cdp = "https://data.cooperationdatabank.org/vocab/prop/"
        output = pipeline(
            model=model, random_seed=random_seed,
            training=self.triples["train"], testing=self.triples["val"],
            model_kwargs={"embedding_dim": embedding_dim},
            optimizer_kwargs={"lr": lr},
            negative_sampler='basic',
            negative_sampler_kwargs = {
                "num_negs_per_pos": num_negs_per_pos,
            },
            epochs=epochs,
            device="cuda:1",
            )
        return output


if __name__ == '__main__':
    FOLDER_IN = "./test_bnlp"
    TH, ES = "regular", "d"
    BN_LP = BNLinkPredictor(dr=os.path.join(FOLDER_IN, f"h_{TH}_es_{ES}_random.csv"),
                            de=os.path.join(FOLDER_IN, f"h_{TH}_es_{ES}_effect.csv"),
                            th=TH,
                            tvt_reg=[0.8, 0.1, 0.1],
                            tvt_effect=[0.8, 0.1, 0.1],
                            )
    PIPELINE = BN_LP.init_hp_pipeline(
                    model='complex',
                    # random_seed=23,
                    epochs=300,
                    embedding_dim=208,
                    lr=0.002,
                    num_negs_per_pos=31
                    )
    METRICS = ['hits@1', 'hits@3', 'hits@10', 'mean_reciprocal_rank']
    for M in METRICS:
        print(f"{M}: {PIPELINE.metric_results.get_metric(M)}")
    PIPELINE.save_to_directory("./test_bnlp/model")
