# -*- coding: utf-8 -*-
"""
Retrieving data for treatment 1 vs. treatment 2 comparison

Conditions:
- Treatment 1 and 2 should have the same generic variable
- Treatment 1 and 2 can have either
    - The same specific independent variable --> OPTION_GROUPING = "same_siv"
    - A different specific independent variable --> OPTION_GROUPING = "different_siv"
"""
import os
from itertools import combinations
import click
from tqdm import tqdm
import pandas as pd
from kglab.helpers.data_load import read_csv
from kglab.helpers.kg_query import run_query
from kglab.helpers.variables import HEADERS_CSV
from src.pipeline import Pipeline
from src.settings import ROOT_PATH

SPARQL_ENDPOINT = "http://localhost:7200/repositories/coda"
OPTION_GROUPING = "different_siv"

QUERY = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX cdo: <https://data.cooperationdatabank.org/vocab/class/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
select ?generic_var ?spec_var_label (group_concat(distinct str(?spec_var_val_label);separator='|') as ?spec_var_val_labels) where { 
    ?spec_var_val rdf:type ?spec_var ;
                  rdfs:label ?spec_var_val_label .
	?spec_var rdfs:subClassOf cdo:IndependentVariable, ?generic_var ;
              rdfs:label ?spec_var_label .
    ?generic_var rdfs:label ?generic_label .
    FILTER(STRENDS(?generic_label, "independent variables"))
} 
GROUP BY ?generic_var ?spec_var_label
"""

def arrange_pair(pair):
    """ Re-ordering pair st. the first SIV before the second one (alphabetical order) """
    if len(pair[0]) == 2:  #same gv
        if pair[0][0] <= pair[1][0]:
            return (pair[0][0], pair[0][1], pair[1][0], pair[1][1])
        return (pair[1][0], pair[1][1], pair[0][0], pair[0][1])

    # different gv
    if pair[0][1] <= pair[1][2]:
        return (pair[0][0], pair[0][1], pair[0][2], pair[1][0], pair[1][1], pair[1][2])
    return (pair[1][0], pair[1][1], pair[1][2], pair[0][0], pair[0][1], pair[0][2])


def get_df_out_pairs(data, option_grouping):
    """ Building df with all combinations of unique (siv1, sivv1, siv2, sivv2) """
    df_out = pd.DataFrame(columns=["generic1", "siv1", "sivv1", "generic2", "siv2", "sivv2"])
    if option_grouping == "same_siv":
        for _, row in tqdm(data.iterrows(), total=len(data)):
            siv = row.spec_var_label.lower()
            sivvs = [x.lower() for x in row.spec_var_val_labels.split("|")]
            pairs = list(combinations(sivvs, 2))
            for (sivv1, sivv2) in pairs:
                df_out.loc[len(df_out)] = [row.generic_var, siv, sivv1, row.generic_var, siv, sivv2]

    if option_grouping == "different_siv":
        for gv, grouped in tqdm(data.groupby("generic_var")):
            siv_sivv_pairs = [(row.spec_var_label.lower(), x.lower()) \
                for _, row in grouped.iterrows() \
                    for x in row.spec_var_val_labels.split("|")]
            pairs = list(combinations(siv_sivv_pairs, 2))
            pairs = list(set(arrange_pair(pair) for pair in pairs))
            for (siv1, sivv1, siv2, sivv2) in pairs:
                df_out.loc[len(df_out)] = [gv, siv1, sivv1, gv, siv2, sivv2]

    if option_grouping == "different_and_same_gv":
        siv_sivv_pairs = [(row.generic_var, row.spec_var_label.lower(), x.lower()) \
            for _, row in data.iterrows() \
                for x in row.spec_var_val_labels.split("|")]
        pairs = list(combinations(siv_sivv_pairs, 2))
        pairs = list(set(arrange_pair(pair) for pair in pairs))
        for (gv1, siv1, sivv1, gv2, siv2, sivv2) in pairs:
            df_out.loc[len(df_out)] = [gv1, siv1, sivv1, gv2, siv2, sivv2]
    return df_out


def add_shape_data(row_, obs_data):
    """ Adding number of data points for statistical analysis """
    pipeline = Pipeline(giv1=row_.generic1, siv1=row_.siv1, sivv1=row_.sivv1,
                        giv2=row_.generic2, siv2=row_.siv2, sivv2=row_.sivv2)
    data = pipeline.get_data_meta_analysis(data=obs_data)
    row_["nb"] = data.shape[0]
    return row_

@click.command()
@click.argument("generic_specific")
@click.argument("obs_data")
@click.argument("option_grouping")
def main(generic_specific, obs_data, option_grouping):
    """ Main """
    if not os.path.exists(generic_specific):
        df = run_query(query=QUERY, sparql_endpoint=SPARQL_ENDPOINT, headers=HEADERS_CSV)
        df.to_csv(generic_specific)
    else:
        df = read_csv(generic_specific)

    obs_data = read_csv(obs_data)

    df_out_path = os.path.join(ROOT_PATH, f"data/same_gv_{option_grouping}_treat_1_2.csv")
    if not os.path.exists(df_out_path):
        df_out = get_df_out_pairs(data=df, option_grouping=option_grouping)
        tqdm.pandas()
        df_out = df_out.progress_apply(lambda row: add_shape_data(row, obs_data), axis=1)
        df_out.to_csv(df_out_path)
    else:
        df_out = read_csv(df_out_path)

    analysis_path = os.path.join(ROOT_PATH, f"data/analysis_{option_grouping}_treat_1_2.csv")
    cols_res = ["k", "b", "se", "zval", "pval", "ci.lb", "ci.ub", "tau2"]
    if not os.path.exists(analysis_path):
        df_out = df_out[df_out.nb >= 10]
        res = []
        for _, row_ in tqdm(df_out.iterrows(), total=len(df_out)):
            pipeline = Pipeline(giv1=row_.generic1, siv1=row_.siv1, sivv1=row_.sivv1,
                                giv2=row_.generic2, siv2=row_.siv2, sivv2=row_.sivv2)
            try:
                res.append(pipeline(data=obs_data))
            except Exception as _:
                res.append({k: None for k in cols_res})
        for col in cols_res:
            df_out[col] = [x[col] for x in res]
        df_out.to_csv(analysis_path)


if __name__ == '__main__':
    # Examples of command (from root directory)
    # python src/get_generic_specific.py data/generic_specific.csv \
    #  data/observationData.csv different_siv
    # python src/get_generic_specific.py data/generic_specific.csv \
    #   data/observationData.csv same_siv
    # python src/get_generic_specific.py data/generic_specific.csv \
    #   data/observationData.csv different_and_same_gv
    if not os.path.exists("data"):
        os.makedirs("data")
    main()
