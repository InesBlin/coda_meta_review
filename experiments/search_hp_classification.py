# -*- coding: utf-8 -*-
"""
Search hyperparameters for classification task
"""
import os
import json
import click
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PARAM_GRID = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 5, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2', 0.5, 0.7],
    'max_leaf_nodes': [None, 10, 30],
    'random_state': [23]
}
PARAMS = list(ParameterGrid(PARAM_GRID))
# N = 300
# random.seed(23)
# PARAMS = random.sample(PARAMS, N)

METRICS = ["acc_train", "acc_val"] + \
    [f"{x}_{y}_{z}" for x in ["precision", "recall", "f1"] \
        for y in ["macro", "micro", "weighted"] \
            for z in ["train", "val"]]
COLUMNS = list(sorted(PARAM_GRID.keys())) + METRICS

def run_one_tree(X_train, y_train, config):
    """ Fit one model """
    clf = DecisionTreeClassifier(
        criterion=config['criterion'],
        splitter=config['splitter'],
        max_depth=config['max_depth'],
        min_samples_split=config['min_samples_split'],
        max_features=config['max_features'],
        random_state=config['random_state'],
    )
    clf.fit(X_train, y_train)
    return clf

def get_metrics(clf, X, y, td):
    """ Get metrics as dict """
    y_pred = clf.predict(X)

    results = {f"acc_{td}": accuracy_score(y, y_pred)}
    for avg in ['macro', 'micro', 'weighted']:
        for (metric, label) in [
            (precision_score, "precision"), (recall_score, "recall"),
            (f1_score, "f1")]:
            results.update({f"{label}_{avg}_{td}": metric(y, y_pred, average=avg)})
    return results

def split_data(df_data, X, y):
    """ Custom split data. Even distribution across train/val/test per GIV.
    Also updating the original data to get the type of data for each row """
    col_td = ["train", "val", "test"]
    splitted_data = {
        x: {"X": [], "y": []} for x in ["train", "val", "test"]
    }
    df_data["td"] = None
    for giv in df_data.giv_prop.unique():
        indexes = list(df_data[df_data.giv_prop == giv].index)
        curr_X = X[indexes]
        curr_y = y[indexes]
        # curr_indexes = np.arange(len(indexes))
        if curr_X.shape[0] < 3:  # random across train/val/test
            random.seed(23)
            shuffled_td = random.sample(col_td, len(col_td))
            random.seed(23)
            sampled = random.sample(shuffled_td, curr_X.shape[0])
            for i, td in enumerate(sampled):
                splitted_data[td]["X"].append(curr_X[i].reshape(1, curr_X[i].shape[0]))
                splitted_data[td]["y"].append(curr_y[i])
                df_data.loc[indexes[i], "td"] = td  # update type of data in df
        elif curr_X.shape[0] == 3:  # one in each
            random.seed(23)
            shuffled_td = random.sample(col_td, len(col_td))
            for i, td in enumerate(shuffled_td):
                splitted_data[td]["X"].append(curr_X[i].reshape(1, curr_X[i].shape[0]))
                splitted_data[td]["y"].append(curr_y[i])
                df_data.loc[indexes[i], "td"] = td  # update type of data in df
        else:
            test_size = 0.2 if curr_X.shape[0] >= 10 else 0.5
            print(curr_X.shape[0])
            curr_X_train, curr_X_, curr_y_train, curr_y_, indexes_train, indexes_ = \
                train_test_split(curr_X, curr_y, indexes,  test_size=test_size, random_state=23)
            curr_X_val, _, curr_y_val, _, indexes_val, indexes_test = \
                train_test_split(curr_X_, curr_y_, indexes_,  test_size=0.5, random_state=23)
            # update numerical data
            splitted_data["train"]["X"].append(curr_X_train)
            if len(curr_X_val.shape) == 1:  # only one sample
                splitted_data["val"]["X"].append(curr_X_val.reshape(1, curr_X[i].shape[0]))
            else:
                splitted_data["val"]["X"].append(curr_X_val)
            splitted_data["train"]["y"] += curr_y_train.tolist()
            splitted_data["val"]["y"] += curr_y_val.tolist()
            # update type of data in df
            df_data.loc[indexes_train, "td"] = "train"
            df_data.loc[indexes_val, "td"] = "val"
            df_data.loc[indexes_test, "td"] = "test"
    return np.concatenate(splitted_data["train"]["X"]), np.concatenate(splitted_data["val"]["X"]), \
        splitted_data["train"]["y"], splitted_data["val"]["y"], df_data



def run_one_config(X_train, y_train, X_val, y_val, config):
    """ Run for one data + one config """
    clf = run_one_tree(X_train, y_train, config)

    results = {}
    results.update(get_metrics(clf, X_train, y_train, "train"))
    results.update(get_metrics(clf, X_val, y_val, "val"))
    return results


@click.command()
@click.argument('folder_data')
@click.argument('folder_embed')
@click.argument('folder_out')
def main(folder_data, folder_embed, folder_out):
    files = [x.replace(".csv", "") for x in os.listdir(folder_data)]
    # to-remove
    files = [x for x in files if 'study_mod' not in x]
    
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    
    for file in tqdm(files):
        logger.info(f"Running file {file}")
        save_path = os.path.join(folder_out, f"{file}.json")
        if os.path.exists(save_path):
            with open(save_path, 'r', encoding='utf-8') as openfile:
                results = json.load(openfile)
        else:
            results = []
        
        df_data = pd.read_csv(os.path.join(folder_data, f"{file}.csv"), index_col=0).reset_index(drop=True)
        X = np.load(os.path.join(folder_embed, f"{file}_x.npy"))
        y = np.load(os.path.join(folder_embed, f"{file}_y.npy"))
        y = y.reshape(y.shape[1])
        X_train, X_val, y_train, y_val, df_data = split_data(df_data=df_data, X=X, y=y)
        save_data_path = os.path.join(folder_out, f"{file}_data.csv")
        if not os.path.exists(save_data_path):
            df_data.to_csv(save_data_path)
        
        exp_run = [{k: x[k] for k in PARAM_GRID.keys()} for x in results]
        params = [x for x in PARAMS if x not in exp_run]
        logger.info(f"{len(PARAMS)} experiments to run in total")
        logger.info(f"{len(params)} to run | {len(exp_run)} already run")
        for config in tqdm(params):
            metrics = run_one_config(X_train, y_train, X_val, y_val, config)
            config.update(metrics)
            results.append(config)

            with open(save_path, 'w', encoding='utf-8') as json_file:
                json.dump(results, json_file, indent=4)
            
            df = pd.DataFrame(results, columns=COLUMNS)
            df.to_csv(os.path.join(folder_out, f"{file}_metrics.csv"))


if __name__ == '__main__':
    # python experiments/search_hp_classification.py ./data/hypotheses/classification/ ./data/hypotheses/embeds ./experiments/classification/hp_search
    main()
