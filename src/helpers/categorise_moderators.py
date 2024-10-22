# -*- coding: utf-8 -*-
"""
Categorising moderators

Script to categorise moderators, assuming that all moderators were already cached 
and saved (cf script `src/helpers/cache_data.py`)

Recap
- country moderators -> all numerical
"""
import json
import click
import pandas as pd
from tqdm import tqdm
from kglab.helpers.kg_query import run_query
from kglab.helpers.variables import HEADERS_CSV

def categorise_country_mod(csv_path: str):
    """ Country moderators: all numerical """
    country_mods = pd.read_csv(csv_path, index_col=0)
    return {mod: "numerical" for mod in country_mods.p.unique()}


def helper_study_mod(x: str):
    """ Rule-based system to determine whether the moderator is numerical or categorical """
    class_start_iri = "https://data.cooperationdatabank.org/vocab/class/"
    if x.startswith(class_start_iri) or "boolean" in x:
        return 'categorical'
    return 'numerical'


def categorise_study_mod(csv_path: str, endpoint: str):
    """ Study moderators: rule-based on the moderator property range """
    study_mods = pd.read_csv(csv_path, index_col=0)
    labels = ", ".join(f'"{x}"' for x in study_mods.label.unique())
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?mod ?range WHERE {
        ?mod rdfs:label ?label ;
             rdfs:range ?range .
        FILTER(?label IN (""" + labels + """))
    }
    """
    df = run_query(query=query, sparql_endpoint=endpoint, headers=HEADERS_CSV)
    return {row["mod"]: helper_study_mod(x=row["range"]) for _, row in df.iterrows()}


def categorise_variable_mod(csv_path: str, endpoint: str):
    """ Variable moderators: rule-based on the moderator property range """
    res = {}
    values = []
    var_mods = pd.read_csv(csv_path, index_col=0)
    mods = var_mods.siv.unique()
    for mod in tqdm(mods):
        query = """
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT DISTINCT ?mod ?range WHERE {
                ?mod rdfs:range ?range .
                FILTER(?mod IN (""" + f'<{mod}>' + """))
            }
            """
        df = run_query(query=query, sparql_endpoint=endpoint, headers=HEADERS_CSV)
        values += list(df.range.unique())
        res.update({row["mod"]: helper_study_mod(x=row["range"]) for _, row in df.iterrows()})
    return res


@click.command()
@click.argument('country')
@click.argument('study')
@click.argument('variable')
@click.argument('endpoint')
@click.argument('save')
def main(country, study, variable, endpoint, save):
    """ Categorising all moderators """
    output = {}
    output.update(
        categorise_country_mod(csv_path=country))
    output.update(
        categorise_study_mod(csv_path=study, endpoint=endpoint))
    output.update(
        categorise_variable_mod(csv_path=variable, endpoint=endpoint))

    with open(save, "w", encoding="utf-8") as openfile:
        json.dump(output, openfile, indent=4)




if __name__ == '__main__':
    # python src/helpers/categorise_moderators.py data/moderators/country_moderators.csv \
    #   data/moderators/study_moderators.csv data/moderators/variable_moderators.csv \
    #   http://localhost:7200/repositories/coda data/cat_moderators.json
    main()
