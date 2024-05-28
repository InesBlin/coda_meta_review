# -*- coding: utf-8 -*-
"""
Search hyperparameters for classification task
"""
import os
import click
from sklearn.model_selection import ParameterGrid
from src.lp.classification import HypothesesClassifier

CV = 5
PARAM_GRID = {
    'learning_rate':[0.001, 0.01, 0.1],
    'n_estimators':[10000, 1000],
    'max_depth':[6, -1],
    'num_leaves':[40, 60, 65, 70],#, 128, 256, 512],
    'colsample_bytree':[0.7, 1],
}
PARAMS = list(ParameterGrid(PARAM_GRID))

@click.command()
@click.argument("data_path")
@click.argument("folder_embeddings")
@click.argument("results_path")
@click.argument("results_csv")
def main(data_path, folder_embeddings, results_path, results_csv):
    """ Main HP search for one file """
    filename = data_path.split("/")[-1].replace(".csv", "")

    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as openfile:
            results = json.load(openfile)
    else:
        results = []
    exp_run = [{k: x[k] for k in PARAM_GRID.keys()} for x in results]
    params = [x for x in PARAMS if x not in exp_run]

    for config in params:
        config_gs = {k: [v] for k, v in config.items()}

        hc = HypothesesClassifier(
            data_path=data_path, 
            X_path=os.path.join(folder_embeddings, f"{filename}_x.npy"),
            y_path=os.path.join(folder_embeddings, f"{filename}_y.npy")
        )
        X_train, _, y_train, _, _, _ = HC.train_test_split()
        gs, model = hc.run_grid_search(param_grid=config_gs, X_train=X_train, y_train=y_train, cv=CV)

# python experiments/search_hp_classification.py ./data/hypotheses/entry/h_regular_es_r.csv ./data/hypotheses/embeds/
00


# MODEL = HC.model
# MODEL.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
print(HC.get_metrics(y_predicted=y_pred_train, y_true=y_train))
print(HC.get_metrics(y_predicted=y_pred_test, y_true=y_test))