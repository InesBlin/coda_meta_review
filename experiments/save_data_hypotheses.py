# -*- coding: utf-8 -*-
"""
Saving entry data for hypotheses generation.

Can then be re-used for:
- LLM prompting,
- Classification task
- LP task

"""
import os
import json
import click
from loguru import logger
from src.lp.compute_hypotheses import HypothesesBuilder

TYPE_HYPOTHESIS = ['regular', 'var_mod', 'study_mod']
ES_MEASURE = ['d']

@click.command()
@click.argument("sparql_endpoint")
@click.argument("mod_to_category")
@click.argument("save_folder")
def main(sparql_endpoint, mod_to_category, save_folder):
    """ Retrieving data for all hypothesis type and es measures """
    with open(mod_to_category, 'r', encoding="utf-8") as openfile:
        mod_to_category = json.load(openfile)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for th in TYPE_HYPOTHESIS:
        for esm in ES_MEASURE:
            hb = HypothesesBuilder(type_h=th, es_measure=esm, mod_to_category=mod_to_category)
            logger.info(f"Fetching info for hypothesis `{th}` with effect size measure `{esm}`")
            save_path = os.path.join(save_folder, f"h_{th}_es_{esm}.csv")
            if not os.path.exists(save_path):
                hb = HypothesesBuilder(type_h=th, es_measure=esm, mod_to_category=mod_to_category)
                obs = hb(sparql_endpoint=sparql_endpoint)
                obs.to_csv(save_path)


if __name__ == '__main__':
    # python experiments/save_data_hypotheses.py http://localhost:7200/repositories/coda cat_moderators.json data/hypotheses/entry
    main()
