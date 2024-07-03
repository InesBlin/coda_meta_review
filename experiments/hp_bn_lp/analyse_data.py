# -*- coding: utf-8 -*-
"""
Analysing results from LP with blank nodes
"""
import os
import numpy as np
import pandas as pd

FOLDER = "./experiments/hp_bn_lp"
TH = ["regular", "study_mod", "var_mod"]
MODELS = ["transe", "distmult", "complex", "rgcn"]

COL_LABEL = [
    ("embedding_dim", "Embedding Dim."),
    ("lr", "Learning rate"),
    ("num_negs_per_pos", "\\# of negatives"),
    ("epochs", "\\# of epochs"),
    ("hits@1", "Hits@1"),
    ("hits@3", "Hits@3"),
    ("hits@10", "Hits@10"),
    ("mean_reciprocal_rank", "MRR"),
]

config_logs = []
logs = {}

for th in TH:
    print(f"====={th}====")
    df = pd.read_csv(os.path.join(FOLDER, f"h_{th}_es_d_hp_100.csv"), index_col=0)[:40]
    configs = df.groupby("model").agg({"embedding_dim": "count"}).reset_index()
    for m in MODELS:
        config_logs.append(str(configs[configs.model == m].embedding_dim.values[0]))
    config_l = "\\# configs & " + ' & '.join(config_logs) + "\\\\"

for th in TH:
    df = pd.read_csv(os.path.join(FOLDER, f"h_{th}_es_d_hp_100.csv"), index_col=0)[:40]
    for m in MODELS:
        curr_df = df[df.model == m]
        best = curr_df[curr_df.mean_reciprocal_rank==np.max(curr_df.mean_reciprocal_rank.values)]
        for col, label in COL_LABEL:
            if col not in logs:
                logs[col] = [label]
            logs[col].append(f"{round(best[col].values[0], 2):,}")
            
            


print(config_l)
for col, _ in COL_LABEL:
    print(" & ".join(logs[col]) + "\\\\")