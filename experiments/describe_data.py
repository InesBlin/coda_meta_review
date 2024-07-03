# -*- coding: utf-8 -*-
"""
Automatic data description for the paper
"""
import os
import pandas as pd
import plotly.express as px

ES_MEASURES = ["d"]
LABELS = ["regular", "study_mod", "var_mod", ]
TD = ["train", "val", "test"]

FOLDER_ENTRY = os.path.join('data', 'hypotheses', 'entry')

RES_ENTRY = []
for th in LABELS:
    for es in ES_MEASURES:
        data = pd.read_csv(os.path.join(FOLDER_ENTRY, f"h_{th}_es_{es}.csv"), index_col=0)
        RES_ENTRY.append((th, es, data.shape[0]))
print("The \\texttt{SPARQL} queries used to get the observations resulted in " + \
    f"{RES_ENTRY[0][2]:,}, {RES_ENTRY[1][2]:,} and {RES_ENTRY[2][2]:,} observations " + \
        "for the regular, study moderator and variable moderator hypotheses respectively.")

RES_CLASSIFICATION_SHAPE = []
for th in LABELS:
    folder = f"./experiments/classification/final/h_{th}_es_d"
    df = pd.read_csv(os.path.join(folder, "data.csv"))
    RES_CLASSIFICATION_SHAPE.extend([f"{df[df.td==x].shape[0]:,}" for x in TD])
print("Classificationd data size")
print(" & ".join(RES_CLASSIFICATION_SHAPE))



labels = {
    "regular": {
    0: "giv", 1: "siv1", 2: "cat1", 3: "siv2", 4: "cat2", 5: "dep"},
    "study_mod": {
    0: "giv", 1: "siv1", 2: "cat1", 3: "siv2", 4: "cat2", 5: "mod", 6: "mod_val", 7: "dep"},
    "var_mod": {
    0: "giv", 1: "siv1", 2: "cat1", 3: "siv2", 4: "cat2", 5: "mod", 6: "mod1", 7: "mod2", 8: "dep"}
}
all_cols = ["giv", "siv1", "cat1", "siv2", "cat2", "mod", "mod_val", "mod1", "mod2", "dep"]

features = {
    th: pd.read_csv(f"./experiments/classification/final/h_{th}_es_d/imp_feature.csv", index_col=0) for th in ["regular", "study_mod", "var_mod"]
}
columns = ["grouped_feat", "imp", "th"]
for th, info in features.items():
    info["th"] = th
    info["grouped_feat"] = info["grouped_feat"].apply(lambda x: labels[th][x])
    feats = list(set(all_cols).difference(set(info.grouped_feat.unique())))
    info = pd.concat([info, pd.DataFrame([(feat, 0, th) for feat in feats], columns=columns)])
    features[th] = info

fig = px.histogram(pd.concat([info for _, info in features.items()]),
                   x="grouped_feat", y="imp", color="th")
fig.update_layout(barmode='group')
fig.update_yaxes(range=[0, 1])
fig.update_xaxes(title_text='Feature')
fig.update_yaxes(title_text='Importance Feature')
fig.write_image(f"./experiments/visualisation/feat_importance_classification.pdf", format='pdf')

from src.lp.blank_node_lp import BNLinkPredictor
folder_in = "data/hypotheses/anyburl"
td = ["train", "val", "test"]
for label in LABELS:
    data = []
    for t in td:
        data.append(pd.read_csv(os.path.join(folder_in, f"{t}_h_{label}_es_d.csv")).shape[0])
    print(label + "\t" + ' & '.join([f"{x:,}" for x in data]) + "\\\\")