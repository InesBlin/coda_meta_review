# -*- coding: utf-8 -*-
"""
generic helpers
"""
import io
import requests
import pandas as pd

def evaluate_list(input_, criteria):
    """ returns boolean vector of whether a criteria is
    in each element of datalist """
    def helper(x):
        return criteria in x.lower() if isinstance(x, str) else False
    if isinstance(input_, list):
        return [helper(x) for x in input_]
    # Else a string
    return helper(input_)

def select_observations(data, siv, sivv, treatment_number):
    """ specific applications of evaluate_list """
    treatment_selection = f"{siv} : {sivv}"
    treatment_value_key = f"treatmentValue{treatment_number}"
    return evaluate_list(list(data[treatment_value_key]), treatment_selection)

def run_request(uri: str, headers: dict):
    """ API call """
    response = requests.get(uri, headers=headers, timeout=3600)

    if "csv" in headers["Accept"]:
        return pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    # Assuming json
    return pd.read_json(io.StringIO(response.content.decode('utf-8')))

def remove_url(url):
    return str(url).split('/')[-1]


def rdflib_to_pd(graph):
    """ Rdflib graph to pandas df with columns ["subject", "predicate", "object"] """
    df = pd.DataFrame(columns=['subject', 'predicate', 'object'])
    for subj, pred, obj in graph:
        df.loc[df.shape[0]] = [str(subj), str(pred), str(obj)]
    return df