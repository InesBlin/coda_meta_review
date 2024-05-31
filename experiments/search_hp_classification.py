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

CV = 5
PARAM_GRID = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5, 0.7],
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


def run_one_config(folder, file, config):
    """ Run for one data + one config """
    X = np.load(os.path.join(folder, f"{file}_x.npy"))
    y = np.load(os.path.join(folder, f"{file}_y.npy"))
    y = y.reshape(y.shape[1])

    X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.2, random_state=23)
    X_val, _, y_val, _ = train_test_split(X_, y_, test_size=0.5, random_state=23)
    clf = run_one_tree(X_train, y_train, config)

    results = {}
    results.update(get_metrics(clf, X_train, y_train, "train"))
    results.update(get_metrics(clf, X_val, y_val, "val"))
    return results


@click.command()
@click.argument('folder_in')
@click.argument('folder_out')
def main(folder_in, folder_out):
    files = list(set('_'.join(x.split('_')[:-1]) for x in os.listdir(folder_in)))
    
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
        
        exp_run = [{k: x[k] for k in PARAM_GRID.keys()} for x in results]
        params = [x for x in PARAMS if x not in exp_run]
        logger.info(f"{len(PARAMS)} experiments to run in total")
        logger.info(f"{len(params)} to run | {len(exp_run)} already run")
        for config in tqdm(params):
            metrics = run_one_config(folder=folder_in, file=file, config=config)
            config.update(metrics)
            results.append(config)

            with open(save_path, 'w', encoding='utf-8') as json_file:
                json.dump(results, json_file, indent=4)
            
            df = pd.DataFrame(results, columns=COLUMNS)
            df.to_csv(os.path.join(folder_out, f"{file}.csv"))
    


if __name__ == '__main__':
    main()
