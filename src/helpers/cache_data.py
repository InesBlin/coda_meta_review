# -*- coding: utf-8 -*-
"""
Caching diverse type of data for the moderators
"""
import os
import click
from kglab.helpers.kg_query import run_query
from kglab.helpers.variables import HEADERS_CSV
from src.helpers.helpers import run_request
from src.moderator import ModeratorComponent
from src.helpers.sparql_queries import SIMPLE_COUNTRY_MOD_QUERY, \
    COMPLEX_COUNTRY_MOD_QUERY, VARIABLE_MOD_QUERY

MODERATOR_C = ModeratorComponent()

@click.command()
@click.argument("type_cache")
@click.argument("save_folder")
def main(type_cache, save_folder):
    """ Running various functions dependending on `type_cache` """
    if type_cache == "study_moderator":
        df = run_request(
            MODERATOR_C.study_moderator_query,
            headers=HEADERS_CSV
        )
        df.to_csv(os.path.join(save_folder, "study_moderators.csv"))

    if type_cache == "country_moderator":
        # All country moderators
        df = run_query(
            query=MODERATOR_C.country_prop_query,
            sparql_endpoint=MODERATOR_C.sparql_endpoint,
            headers=HEADERS_CSV
        )
        df.to_csv(os.path.join(save_folder, "country_moderators.csv"))

        df = run_query(
            query=SIMPLE_COUNTRY_MOD_QUERY,
            sparql_endpoint=MODERATOR_C.sparql_endpoint,
            headers=HEADERS_CSV
        )
        df.to_csv(os.path.join(save_folder, "simple_country_moderators.csv"))

        df = run_query(
            query=COMPLEX_COUNTRY_MOD_QUERY,
            sparql_endpoint=MODERATOR_C.sparql_endpoint,
            headers=HEADERS_CSV
        )
        df.to_csv(os.path.join(save_folder, "complex_country_moderators.csv"))

    if type_cache == "variable_moderator":
        df = run_query(
            query=VARIABLE_MOD_QUERY,
            sparql_endpoint=MODERATOR_C.sparql_endpoint,
            headers=HEADERS_CSV
        )
        df.to_csv(os.path.join(save_folder, "variable_moderators.csv"))

    return


if __name__ == '__main__':
    # python src/cache_data.py study_moderator data/moderators
    # python src/cache_data.py country_moderator data/moderators
    # python src/cache_data.py variable_moderator data/moderators
    main()
