# -*- coding: utf-8 -*-
"""
Running diverse set of experiments
"""
import pickle
from typing import Union
import click
import pandas as pd
from tqdm import tqdm
from kglab.helpers.data_load import read_csv
from kglab.helpers.kg_query import run_query
from kglab.helpers.variables import HEADERS_CSV
from src.pipeline import Pipeline

SPARQL_ENDPOINT = "http://localhost:7200/repositories/coda"
QUERY_LABEL = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
select * where { 
	?node rdfs:label ?label ;
       rdf:type rdf:Property .
    FILTER (LCASE(str(?label)) = "<label>") 
} 
"""
CACHED = {
        "study_moderators": "./data/moderators/study_moderators.csv",
        "country_moderators": "./data/moderators/country_moderators.csv",
        "simple_country_moderators": "./data/moderators/simple_country_moderators.csv",
        "complex_country_moderators": "./data/moderators/complex_country_moderators.csv",
        "variable_moderators": "./data/moderators/variable_moderators.csv"
    }

def add_siv_id(content):
    """ Retrieve node from SIV """
    res = run_query(query=QUERY_LABEL.replace("<label>", content),
                    sparql_endpoint=SPARQL_ENDPOINT,
                    headers=HEADERS_CSV).node.values[0]
    return res.split("/")[-1]


def update_grouping(row):
    """ Add info for moderator """
    row["giv1_id"] = row["generic1"].split("/")[-1]
    row["giv2_id"] = row["generic2"].split("/")[-1]
    row["siv1_id"] = add_siv_id(row["siv1"])
    row["siv2_id"] = add_siv_id(row["siv2"])
    return row


def get_moderators(pipeline, data_run, info, type_mod):
    """ Retrieving list of moderators """
    if type_mod == "variable":
        moderators = pipeline.meta_analysis.moderator \
                .get_variable_moderators(
                    data=data_run, info=info)
        return list(moderators)

    if type_mod == "study":
        moderators = pipeline.meta_analysis.moderator.get_study_moderators()
        return list(moderators.keys())

    #mod == "country"
    moderators, _, _ = pipeline.meta_analysis.moderator.get_country_moderators()
    return list(moderators.pLabel.astype(str).str.lower())


def run_meta_analysis_with_moderators(groupings: pd.DataFrame, data: pd.DataFrame,
                                      type_mod: str, cached_moderator: dict,
                                      updated_grouping: Union[str, None] = None):
    """ Run meta-analysis on T1 vs. T2 with moderators 
    ATM: study moderators 
    
    - grouping_path: data with meta-analyses to run
        (eg. from same generic variable, from different ones, etc)
    - data: starting data, typically the one from src/get_obs_data.py 
    - mods: type of moderator to use, must be a in ["variable", "study", "country"]
    - cached_moderator: cached moderator info for more efficiency
    """
    possible_mods = ["variable", "study", "country"]
    if type_mod not in possible_mods:
        raise ValueError(f"Possible `mod` value must be within {possible_mods}")


    if any(x not in groupings.columns for x in ["giv1_id", "giv1_id", "siv1_id", "siv2_id"]):
        tqdm.pandas()
        print("Updating IRIs for moderator selection")
        groupings = groupings.progress_apply(update_grouping, axis=1)
        if updated_grouping:
            groupings.to_csv(updated_grouping)

    groupings = groupings.sort_values(by="nb", ascending=False).reset_index()
    res = {"type_mod": type_mod}
    # groupings = groupings[:2]
    for index, row_ in tqdm(groupings.sort_values(by="nb", ascending=False).iterrows(),
                            total=len(groupings)):
        res[index] = {
            "giv1": row_.giv1_id, "siv1": row_.siv1, "sivv1": row_.sivv1,
            "giv2": row_.giv2_id, "siv2": row_.siv2, "sivv2": row_.sivv2,
            "meta_analyses": {}
        }

        pipeline = Pipeline(
            giv1=row_.giv1_id, siv1=row_.siv1, sivv1=row_.sivv1,
            giv2=row_.giv2_id, siv2=row_.siv2, sivv2=row_.sivv2,
            **cached_moderator)
        data_run = pipeline.get_data_meta_analysis(data=data)
        info={"giv1": pipeline.giv1, "giv2": pipeline.giv2,
              "siv1": row_.siv1_id, "siv2": row_.siv2_id}
        moderators = get_moderators(pipeline=pipeline, data_run=data_run,
                                    info=info,  type_mod=type_mod)

        # var_moderators = pipeline.meta_analysis.moderator \
        #     .get_variable_moderators(
        #         data=data_run, info={"giv1": pipeline.giv1, "giv2": pipeline.giv2,
        #                                 "siv1": row_.siv1_id, "siv2": row_.siv2_id})
        # for index_mod, mod in enumerate(var_moderators):
        # moderators = moderators[:2]
        for index_mod, mod in enumerate(moderators):
            try:
                results_rma, refs = pipeline.meta_analysis(
                    type_rma="uni", es_measure="d", yi="effectSize", data=data_run,
                    method="REML", vi="variance", mods={type_mod: [mod]})
                res[index]["meta_analyses"][index_mod] = {
                    "ref": refs[mod],
                    "moderator": mod,
                    "result_rma": {k: v for k, v in results_rma.items() if k != "data"}
                }
            except Exception as _:
                res[index]["meta_analyses"][index_mod] = {
                    "moderator": mod,
                    # "result_rma": e
                }
    return res


@click.command()
@click.argument("groupings")
@click.argument("data")
@click.argument("type_mod")
@click.argument("save_path")
@click.argument("updated_grouping")
def main(groupings, data, type_mod, save_path, updated_grouping):
    """ Main script for moderators experiment """
    groupings = read_csv(groupings)
    groupings = groupings[groupings.nb >= 3]

    data = read_csv(data)
    res = run_meta_analysis_with_moderators(groupings=groupings, data=data,
                                            cached_moderator=CACHED,
                                            type_mod=type_mod, updated_grouping=updated_grouping)
    with open(save_path, "wb") as file:
        pickle.dump(res, file)


if __name__ == '__main__':
    # VARIABLE
    # python src/experiments.py ./data/same_gv_different_siv_treat_1_2_with_ids.csv \
    # ./data/observationData.csv variable meta_analyses_with_variable_moderators.pkl ./saved.csv
    # STUDY
    # python src/experiments.py ./data/same_gv_different_siv_treat_1_2_with_ids.csv \
    # ./data/observationData.csv study meta_analyses_with_study_moderators.pkl ./saved.csv
    # COUNTRY
    # python src/experiments.py ./data/same_gv_different_siv_treat_1_2_with_ids.csv \
    # ./data/observationData.csv country meta_analyses_with_country_moderators.pkl ./saved.csv
    main()
