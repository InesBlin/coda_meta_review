# -*- coding: utf-8 -*-
"""
Link Prediction for Blank Node
"""
import random
from typing import Union, Tuple
import pandas as pd

def init_empty_df():
    return pd.DataFrame(columns=["subject", "predicate", "object"])

def split_subject(subject: str, th: str):
    """ Split subject into a more readable format """
    if th == "regular":
        return subject.split("_")[0].split('/')[-1]
    
def distribute(indexes, train, val, test):
    """ Extract `train`, `val`and `test` elements from the list of `indexes``
    such that their sum equals the indexes
    """
    train_l = random.sample(indexes, train)
    remaining = list(set(indexes).difference(set(train_l)))

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
    if sum(tvt_split) != 1:
        raise ValueError("The sum of the values in `train_val_test_split` must be 1.")
    if any(x > tvt_split[0] for x in tvt_split[1:]):
        raise ValueError("The split for `training` must be the highest.")

    data["h_id"] = data["subject"].apply(lambda x: split_subject(x, th))
    grouped = data.groupby(["h_id", "predicate"]).agg({"subject": "count"}).reset_index().rename(columns={"subject": "nb"})

    cols_triples = ["subject", "predicate", "object"]
    col_td = ["train", "val", "test"]
    res = {x: init_empty_df() for x in col_td}

    for _, row in grouped.iterrows():
        curr_data = data[(data.predicate==row.predicate) & (data.h_id==row.h_id)]
        curr_data = curr_data[cols_triples].reset_index()
        if row.nb < 3:
            # random.seed(23)
            shuffled_td = random.sample(col_td, len(col_td))
            # random.seed(23)
            sampled = random.sample(shuffled_td, row.nb)
            for index, row_curr_data in curr_data.iterrows():
                td = sampled[index]
                res[td] = pd.concat([res[td], pd.DataFrame([[row_curr_data[x] for x in cols_triples]], columns=cols_triples)])
        
        elif row.nb == 3:
            # random.seed(23)
            sampled = random.sample(col_td, len(col_td))
            for index, row_curr_data in curr_data.iterrows():
                td = sampled[index]
                res[td] = pd.concat([res[td], pd.DataFrame([[row_curr_data[x] for x in cols_triples]], columns=cols_triples)])
        
        else:
            nb_test = max(int(tvt_split[2]*row.nb), 1)
            nb_val = max(int(tvt_split[1]*row.nb), 1)
            nb_train = row.nb - nb_test - nb_val
            train_i, val_i, test_i = distribute(list(curr_data.index), nb_train, nb_val, nb_test)
            for indexes, td in [(train_i, "train"), (val_i, "val"), (test_i, "test")]:
                res[td] = pd.concat([res[td], curr_data[curr_data.index.isin(indexes)]])
            
    return res

# to concatenate triplesfactory datasets, go back to df representations and concatenate
# kg1_train, kg1_val, kg1_test = kg1_factory.split([0.8, 0.1, 0.1])

if __name__ == '__main__':
    DATA = pd.read_csv("./data/hypotheses/bn/h_regular_es_d_effect.csv", index_col=0)
    TRAIN_VAL_TEST = [0.8, 0.1, 0.1]
    TH = "regular"
    split_effect_triples(data=DATA, tvt_split=TRAIN_VAL_TEST, th=TH)