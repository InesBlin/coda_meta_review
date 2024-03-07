# -*- coding: utf-8 -*-
"""
Running diverse set of experiments
"""
import pickle
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


def run_meta_analysis_with_moderators(grouping_path, data):
    """ Run meta-analysis on T1 vs. T2 with moderators 
    ATM: study moderators """
    groupings = read_csv(grouping_path)
    groupings = groupings[groupings.nb >= 3]

    if any(x not in groupings.columns for x in ["giv1_id", "giv1_id", "siv1_id", "siv2_id"]):
        tqdm.pandas()
        print("Updating IRIs for moderator selection")
        groupings = groupings.progress_apply(update_grouping, axis=1)
        groupings.to_csv(f"{grouping_path.replace('.csv', '')}_with_ids.csv")
    
    groupings = groupings.sort_values(by="nb", ascending=False).reset_index()
    res = {}
    for index, row_ in tqdm(groupings.sort_values(by="nb", ascending=False).iterrows(), total=len(groupings)):
        res[index] = {
            "giv1": row_.giv1_id, "siv1": row_.siv1, "sivv1": row_.sivv1,
            "giv2": row_.giv2_id, "siv2": row_.siv2, "sivv2": row_.sivv2,
            "meta_analyses": {}
        }
        if index < 1:
            pipeline = Pipeline(
                giv1=row_.giv1_id, siv1=row_.siv1, sivv1=row_.sivv1,
                giv2=row_.giv2_id, siv2=row_.siv2, sivv2=row_.sivv2)
            data_run = pipeline.get_data_meta_analysis(data=data)

            var_moderators = pipeline.meta_analysis.moderator \
                .get_variable_moderators(
                    data=data_run, info={"giv1": pipeline.giv1, "giv2": pipeline.giv2,
                                         "siv1": row_.siv1_id, "siv2": row_.siv2_id})
            for index_mod, mod in enumerate(var_moderators):
                try:
                    results_rma = pipeline.meta_analysis(
                        type_rma="uni", es_measure="d", yi="effectSize", data=data_run,
                        method="REML", vi="variance", mods={"variable": [mod]})
                    print(test)
                    res[index]["meta_analyses"][index_mod] = {
                        "moderator": mod,
                        "result_rma": {k: v for k, v in results_rma.items() if k != "data"}
                    }
                except Exception as _:
                    res[index]["meta_analyses"][index_mod] = {
                        "moderator": mod,
                        # "result_rma": e
                    }
    return res


if __name__ == '__main__':
    GROUPING_PATH = "./data/same_gv_different_siv_treat_1_2_with_ids.csv"
    DATA = read_csv("./data/observationData.csv")
    SAVE_PATH = "meta_analyses_with_moderators.pkl"
    RES = run_meta_analysis_with_moderators(grouping_path=GROUPING_PATH, data=DATA)

    with open(SAVE_PATH, "wb") as file:
        pickle.dump(RES, file)
