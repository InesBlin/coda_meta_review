# -*- coding: utf-8 -*-
"""
Analysing results from data-based prompts
"""
import click
import pandas as pd
import plotly.express as px

def get_category(row):
    """ Category of hypotheses
    - 2: correct, statistically significant
    - 1: correct, not statistically significant
    - 0: N/A
    - -1: incorrect, not statistically significant
    - -2: incorrect, statistically significant
    """
    if row["k"] == "N/A":
        row["category"] = 0
        return row
    
    if ((row["Higher/Lower"] == "higher" and row["b"] > 0 and row["pval"] < 0.01) or \
        (row["Higher/Lower"] == "lower" and row["b"] < 0 and row["pval"] < 0.01)):
        row["category"] = 2
        return row
    
    if ((row["Higher/Lower"] == "higher" and row["b"] > 0 and row["pval"] >= 0.01) or \
        (row["Higher/Lower"] == "lower" and row["b"] < 0 and row["pval"] >= 0.01)):
        row["category"] = 1
        return row
    
    if ((row["Higher/Lower"] == "higher" and row["b"] < 0 and row["pval"] < 0.01) or \
        (row["Higher/Lower"] == "lower" and row["b"] > 0 and row["pval"] < 0.01)):
        row["category"] = -2
        return row
    
    if ((row["Higher/Lower"] == "lower" and row["b"] > 0 and row["pval"] >= 0.01) or \
        (row["Higher/Lower"] == "higher" and row["b"] < 0 and row["pval"] >= 0.01)):
        row["category"] = -1
        return row
    
    row["category"] = -1 if row["pval"] >= 0.01 else -2
    return row
    

@click.command()
@click.argument("data_path")
def main(data_path):
    """ Main """
    data = pd.read_csv(data_path, index_col=0)
    data = data[~data.k.isna()]
    data = data.apply(get_category, axis=1)
    print(f"# of Hypotheses: {data.shape[0]}")
    print(data.groupby("category").agg({"Templated Hypothesis": "count"}))
    print(data.groupby(["Name", "category"]).agg({"Templated Hypothesis": "count"}))

    color_map = {-2: px.colors.qualitative.Set1[0], -1: px.colors.qualitative.Pastel1[0],
                #  0: px.colors.qualitative.Set1[8],
                 1: px.colors.qualitative.Pastel1[1], 2: px.colors.qualitative.Set1[1]}
    fig = px.histogram(data, x="Name", color="category", barmode="group",
                       category_orders={"category": [-2, -1, 1, 2]}, color_discrete_map=color_map)
    fig.update_traces(opacity=0.7)
    fig.update_layout(bargap=0.5)
    fig.write_html("name_category.html")


if __name__ == '__main__':
    main()