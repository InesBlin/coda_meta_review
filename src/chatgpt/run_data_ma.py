# -*- coding: utf-8 -*-
"""
Generate prompts based on the ontology
"""
import os
import re
from tqdm import tqdm
import click
import pandas as pd
from src.pipeline import Pipeline
from kglab.helpers.data_load import read_csv

DATA = read_csv("./data/observationData.csv")
CACHED = {
        "study_moderators": "./data/moderators/study_moderators.csv",
        "country_moderators": "./data/moderators/country_moderators.csv",
        "simple_country_moderators": "./data/moderators/simple_country_moderators.csv",
        "complex_country_moderators": "./data/moderators/complex_country_moderators.csv",
        "variable_moderators": "./data/moderators/variable_moderators.csv"
    }
REGEX = r"Cooperation is significantly (higher|lower) when (.*) is (.*), compared to when (.* )is (.*)\."

def get_param_info_one_line(text):
    """ Retrieve params with regex """
    matches = re.finditer(REGEX, text, re.MULTILINE)
    for _, match in enumerate(matches, start=1):
        return match.groups()
    return None

def format_info(info, name):
    """ List to dict """
    return {"giv1": name, "siv1": info[1].strip(), "sivv1": info[2].strip(),
            "giv2": name, "siv2": info[3].strip(), "sivv2": info[4].strip(),
            "outcome": info[0].strip()}

def get_params(folder, files):
    """ Retrieve dictionary of params """
    res = {}
    for file in files:
        name = file.replace(".txt", "")
        with open(os.path.join(folder, file), "r", encoding="utf-8") as openfile:
            lines = openfile.readlines()
            lines = [x.replace("\n", "") for x in lines]
        infos = [get_param_info_one_line(line) for line in lines]
        res[name] = [format_info(x, name) for x in infos if x]
    return res

def get_param_description(p):
    """ Param human readable des """
    hypothesis = f"Cooperation is significantly {p['outcome']} when {p['siv1']} is {p['sivv1']} compared to when {p['siv2']} is {p['sivv2']}."
    # return f"{p['giv1']};{hypothesis};{p['outcome']};{p['giv1']};{p['siv1']};{p['sivv1']};{p['giv2']};{p['siv2']};{p['sivv2']}"
    return [p['giv1'], hypothesis, p['outcome'], p['giv1'], p['siv1'], p['sivv1'], p['giv2'], p['siv2'], p['sivv2']]

def run_meta_analyses(params):
    """ Meta-analyses for all """
    df = pd.DataFrame(columns=["Name", "Templated Hypothesis", "Higher/Lower", "GIV1", "SIV1", "SIVV1", "GIV2", "SIV2", "SIVV2", "k", "b", "pval"])
    keys = list(sorted(params.keys()))
    for key in tqdm(keys):
        info = params[key]
    # for _, info in tqdm(params.items()):
        for p in info:
            try:
                pipeline = Pipeline(
                    giv1=p['giv1'], siv1=p['siv1'], sivv1=p['sivv1'],
                    giv2=p['giv2'], siv2=p['siv2'], sivv2=p['sivv2'], **CACHED)
                curr_res, _ = pipeline(data=DATA, mods=None)
                # metrics = ";".join([str(curr_res[x].reshape((1,))[0]).replace(".", ",") for x in ["k", "b", "pval"]])
                metrics = [int(curr_res["k"].reshape((1,))[0]), float(curr_res["b"].reshape((1,))[0]), float(curr_res["pval"].reshape((1,))[0])]
                # final_info.append(get_param_description(p=p) + ";" + metrics)
                df.loc[len(df)] = get_param_description(p=p) + metrics
            except Exception as e:
                print(e)
                # final_info.append(get_param_description(p=p) + ";" + "N/A;N/A;N/A")
                df.loc[len(df)] = get_param_description(p=p) + ["N/A", "N/A", "N/A"]
    # return final_info
    return df

@click.command()
@click.argument("input_folder")
def main(input_folder):
    """ Retrieve params and run meta-analyses """
    hypotheses = os.listdir(input_folder)
    params = get_params(folder=input_folder, files=hypotheses)
    df = run_meta_analyses(params=params)
    df.to_csv("prompt_data_based.csv")


if __name__ == '__main__':
    main()
