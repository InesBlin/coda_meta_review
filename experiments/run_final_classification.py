# -*- coding: utf-8 -*-
"""
Final classification model + hypothesis generation
"""
import os
import json
import click
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.knowledge import generate_hypothesis


ID_TO_CLASSES = {
    0: 'negative', 1: 'noEffect', 2: 'positive'
}
CLASSES_TO_COMPARATIVE = {
    'negative': 'lower', 'positive': 'higher', "noEffect": "n/a"
}

BEST_PARAMS = {
    "h_regular_es_d": {
        "criterion": 'gini',
        "splitter": 'random',
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,  
        "max_features": 'sqrt',
        "random_state": 23  # reproducibility
    },
    # "h_regular_es_r": {
    #     "criterion": 'entropy',
    #     "splitter": 'random',
    #     "max_depth": 15,
    #     "min_samples_split": 2,
    #     "min_samples_leaf": 2,  
    #     "max_features": 0.5,
    #     "random_state": 23  # reproducibility
    # },
    "h_var_mod_es_d": {
        "criterion": 'gini',
        "splitter": 'random',
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 2,  
        "max_features": 0.5,
        "random_state": 23  # reproducibility
    },
    # "h_var_mod_es_r": {
    #     "criterion": 'entropy',
    #     "splitter": 'best',
    #     "max_depth": 5,
    #     "min_samples_split": 5,
    #     "min_samples_leaf": 2,  
    #     "max_features": 'sqrt',
    #     "random_state": 23  # reproducibility
    # },
    # "h_study_mod_es_r": {
    #     "criterion": 'entropy',
    #     "splitter": 'best',
    #     "max_depth": 15,
    #     "min_samples_split": 2,
    #     "min_samples_leaf": 2,  
    #     "max_features": "sqrt",
    #     "random_state": 23  # reproducibility
    # },
    "h_study_mod_es_d": {
        "criterion": 'entropy',
        "splitter": 'random',
        "max_depth": 15,
        "min_samples_split": 10,
        "min_samples_leaf": 2,  
        "max_features": None,
        "random_state": 23  # reproducibility
    }
}



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


def enrich_data(df, clf, th, X, y, unique_y):
    """ Add various columns to the original data """
    df["pred_num"] = clf.predict(X)
    df["pred_readable"] = df["pred_num"].apply(lambda x: ID_TO_CLASSES[x])
    df["comparative"] = df["pred_readable"].apply(lambda x: CLASSES_TO_COMPARATIVE[x])
    df["pred_true"] = y
    df[[f"score_{x}" for x in unique_y]] = clf.predict_proba(X)
    df['max_score'] = df[[f"score_{x}" for x in unique_y]].max(axis=1)

    df = df[df.effect != 'noEffect']
    df = df.apply(lambda row: generate_hypothesis(row, th), axis=1)
    return df


def save_top_hypothesis(df, folder_save, top_n: int = 5):
    """ Saving top scored hypotheses for each giv in the test data
    (only keeping the ones with lower/higher prediction) """
    if not os.path.exists(folder_save):
        os.makedirs(folder_save)
    pred_keep = [0, 2]
    for giv in df.giv_prop.unique():
        name = giv.split("/")[-1]
        curr_df = df[(df.giv_prop == giv) & (df.td == "test") & (df.pred_num.isin(pred_keep))]
        if curr_df.shape[0] > 0:
            curr_df = curr_df.sort_values(by="max_score", ascending=False)
            top = min(curr_df.shape[0], top_n)
            curr_df.to_csv(os.path.join(folder_save, f"{name}.csv"))
            f = open(os.path.join(folder_save, f"{name}.txt"), 'w', encoding='utf-8')
            for _, row in curr_df[:top].iterrows():
                f.write(f"{row.hypothesis}\n")
            f.close()


@click.command()
@click.argument("folder_in")
@click.argument("folder_embed")
@click.argument("folder_out")
def main(folder_in, folder_embed, folder_out):
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    for thd, config in BEST_PARAMS.items():
        th = thd.split('h_')[1].split('_es')[0]
        save_f = os.path.join(folder_out, thd)
        if not os.path.exists(save_f):
            os.makedirs(save_f)
        if not os.path.exists(os.path.join(save_f, "data.csv")):
            df = pd.read_csv(os.path.join(folder_in, f"{thd}_data.csv"), index_col=0).reset_index(drop=True)

            X = np.load(os.path.join(folder_embed, f"{thd}_x.npy"))
            y = np.load(os.path.join(folder_embed, f"{thd}_y.npy"))
            y = y.reshape(y.shape[1])

            X_train = X[list(df[(df.td == "train") & (df.effect != 'noEffect')].index)]
            y_train = [y[i] for i in list(df[(df.td == "train") & (df.effect != 'noEffect')].index)]
            X_test = X[list(df[(df.td == "test") & (df.effect != 'noEffect')].index)]
            y_test = [y[i] for i in list(df[(df.td == "test") & (df.effect != 'noEffect')].index)]

            clf = DecisionTreeClassifier(
                criterion=config['criterion'],
                splitter=config['splitter'],
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split'],
                max_features=config['max_features'],
                random_state=config['random_state'],
            )
            clf.fit(X_train, y_train)

            results = {}
            results.update(get_metrics(clf, X_train, y_train, "train"))
            results.update(get_metrics(clf, X_test, y_test, "test"))
            with open(os.path.join(save_f, "results.json"), 'w', encoding='utf-8') as openfile:
                json.dump(results, openfile, indent=4)

            df = enrich_data(df, clf, th, X, y, set(y_train))
            df.to_csv(os.path.join(save_f, "data.csv"))
            save_top_hypothesis(df=df, folder_save=os.path.join(save_f, "outputs"), top_n=5)



if __name__ == '__main__':
    # python experiments/run_final_classification.py ./experiments/classification/hp_search ./data/hypotheses/embeds ./experiments/classification/final
    main()
