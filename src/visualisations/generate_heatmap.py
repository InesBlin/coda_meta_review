# -*- coding: utf-8 -*-
"""
Generate heatmaps from results
"""
import click
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from kglab.helpers.data_load import read_csv

def get_df_filter(df_input, min_k=10, min_b=-1, max_b=1):
    """ Filter on sample size + effect size """
    df_filter = df_input[(df_input["k"] >= min_k) & \
        (min_b <= df_input["b"]) & \
            (df_input["b"] <= max_b)]
    print(f"DATA filtered shape: {df_filter.shape[0]}")
    sivvs = list(set(list(df_filter.sivv1.unique()) + list(df_filter.sivv2.unique())))
    print(f"Unique sivvs: {len(sivvs)}")
    print(f"# of generic variables: {df_filter.generic.unique().shape[0]}")
    return df_filter

@click.command()
@click.argument("data")
@click.argument("save_html")
def main(data, save_html):
    """ Main """
    df = read_csv(data)
    df = df[~df.k.isna()]
    df.generic = df.generic.apply(lambda x: x.split("/")[-1].replace("Variable", ""))
    print(f"Data shape: {df.shape[0]}")
    df_filter = get_df_filter(df_input=df)


    curr_df = df_filter.copy()
    with pd.option_context('mode.chained_assignment', None):
        for i in ["1", "2"]:
            curr_df[f"label{i}"] = curr_df[f"generic{i}"] + " -> " + \
                curr_df[f"siv{i}"] + " -> " + curr_df[f"sivv{i}"]
    curr_df = curr_df.sort_values(by=["label1", "label2"])
    labels1 = sorted(curr_df.label1.unique())
    labels1_to_index = {x: i for i, x in enumerate(labels1)}
    labels2 = sorted(curr_df.label2.unique())
    labels2_to_index = {x: i for i, x in enumerate(labels2)}

    data_hm = np.zeros((len(labels1), len(labels2)))
    for _, row in curr_df.iterrows():
        data_hm[labels1_to_index[row.label1]][labels2_to_index[row.label2]] = row.b
    custom_data = np.array([[l1 + " || " + l2 for l2 in labels2] for l1 in labels1])
    heatmap = go.Heatmap(
        z=data_hm, colorscale="rdgy_r", zmin=-1, zmax=1,
        customdata=custom_data,
        hovertemplate = "Treatment 1 vs. 2:<br>%{customdata}<br>Effect Size: %{z}<extra></extra>")
    fig = go.Figure(data=heatmap)
    fig.update_layout(width=800, height=600)
    fig.write_html(save_html)


if __name__ == '__main__':
    main()
