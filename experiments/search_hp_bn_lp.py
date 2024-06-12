# -*- coding: utf-8 -*-
"""
Search hyperparameters for BN LP
"""

import os
import json
import click
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from src.lp.blank_node_lp import BNLinkPredictor

METRICS = ['hits@1', 'hits@3', 'hits@10', 'mean_reciprocal_rank']
ES_MEASURES = ["d", "r"]
LABELS = ["regular", "var_mod"] #, "study_mod"]
PARAM_GRID = {
    "embedding_dim": [x*16 for x in range(1, 33)],
    "lr": [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
    "num_negs_per_pos": [1] + [10*x for x in range(1, 11)],
    "epochs": [100*x for x in range(1, 6)],
    "model": ["rgcn", "distmult", "complex", "transe"]
}
PARAMS = list(ParameterGrid(PARAM_GRID))
N = 100
random.seed(23)
PARAMS = random.sample(PARAMS, N)
COLUMNS = list(sorted(PARAM_GRID.keys())) + METRICS

@click.command()
@click.argument("folder_in")
@click.argument("folder_out")
def main(folder_in, folder_out):
    """ Main, running the grid """
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    for th in LABELS:
        for es in ES_MEASURES:
            results_path = os.path.join(folder_out, f"h_{th}_es_{es}_hp_{N}.json")
            results_csv = os.path.join(folder_out, f"h_{th}_es_{es}_hp_{N}.csv")
            if os.path.exists(results_path):
                with open(results_path, 'r', encoding='utf-8') as openfile:
                    results = json.load(openfile)
            else:
                results = []
            exp_run = [{k: x[k] for k in PARAM_GRID.keys()} for x in results]
            params = [x for x in PARAMS if x not in exp_run]
            print(f"{len(PARAMS)} experiments to run in total")
            print(f"{len(params)} to run | {len(exp_run)} already run")

            bn_lp = BNLinkPredictor(dr=os.path.join(folder_in, f"h_{th}_es_{es}_random.csv"),
                                    de=os.path.join(folder_in, f"h_{th}_es_{es}_effect.csv"),
                                    th=th)
            for config in tqdm(params):
                pipeline = bn_lp.init_hp_pipeline(
                    model=config["model"], random_seed=23, epochs=config["epochs"],
                    embedding_dim=config["embedding_dim"], lr=config["lr"],
                    num_negs_per_pos=config["num_negs_per_pos"])
                config.update({m: pipeline.metric_results.get_metric(m) for m in METRICS})
                results.append(config)
            
                with open(results_path, 'w', encoding='utf-8') as json_file:
                    json.dump(results, json_file, indent=4)
            
                df = pd.DataFrame(results, columns=COLUMNS)
                df.to_csv(results_csv)


if __name__ == '__main__':
    # python experiments/search_hp_bn_lp.py data/hypotheses/bn/ experiments/hp_bn_lp/
    main()
