# -*- coding: utf-8 -*-
"""
Search hyperparameters for KG embeddings
"""
import os
import json
import click
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from src.lp.kg_embedder import KGEmbedder

METRICS = ['hits@1', 'hits@3', 'hits@10', 'mean_reciprocal_rank']

PARAM_GRID = {
    # "embedding_dim": [x*16 for x in range(1, 33)],
    "embedding_dim": [64, 128, 256, 512],
    # "lr": [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
    "lr": [0.001, 0.01, 0.1],
    # "num_negs_per_pos": [1] + [10*x for x in range(1, 11)],
    "num_negs_per_pos": [30, 50, 80],
    "epochs": [100*x for x in range(1, 6)],
    "model": ["rgcn", "distmult"]
}
PARAMS = list(ParameterGrid(PARAM_GRID))
COLUMNS = list(sorted(PARAM_GRID.keys())) + METRICS


@click.command()
@click.argument("data")
@click.argument("subject_col")
@click.argument("predicate_col")
@click.argument("object_col")
@click.argument("results_path")
@click.argument("results_csv")
def main(data, subject_col, predicate_col, object_col, results_path, results_csv):
    """ Main, running the grid """
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as openfile:
            results = json.load(openfile)
    else:
        results = []
    exp_run = [{k: x[k] for k in PARAM_GRID.keys()} for x in results]
    params = [x for x in PARAMS if x not in exp_run]
    kg_emb = KGEmbedder(
        data_path=data,
        spo_cols=[subject_col, predicate_col, object_col])
    for config in tqdm(params):
        pipeline = kg_emb.init_pipeline(
            model=config["model"], random_seed=23, epochs=config["epochs"],
            embedding_dim=config["embedding_dim"], lr=config["lr"],
            num_negs_per_pos=config["num_negs_per_pos"]
        )
        config.update({m: pipeline.metric_results.get_metric(m) for m in METRICS})
        results.append(config)
    
        with open(results_path, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, indent=4)
    
        df = pd.DataFrame(results, columns=COLUMNS)
        df.to_csv(results_csv)


if __name__ == '__main__':
    # python experiments/search_hyperparameters.py ./data/vocab.csv s p o experiments/results_kg_embedding_hp.json experiments/results_kg_embedding_hp.csv
    main()