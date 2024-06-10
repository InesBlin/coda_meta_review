# -*- coding: utf-8 -*-
""" Build n-ary hypotheses from the data 

Adapted from the original code: https://github.com/roosyay/CoDa_Hypotheses

run SPARQL query: 
run_query(
            query=MODERATOR_C.country_prop_query,
            sparql_endpoint=MODERATOR_C.sparql_endpoint,
            headers=HEADERS_CSV
        )
"""
import math
from typing import Union
from tqdm import tqdm
import pandas as pd
import numpy as np
from rdflib import Graph
from loguru import logger
from validator_collection import checkers
from kglab.helpers.variables import HEADERS_CSV, HEADERS_RDF_XML
from kglab.helpers.kg_query import run_query
from src.lp.sparql_queries import HB_REGULAR_T, TREATMENT_VALS_T_REGULAR, \
    TREATMENT_VALS_T_VAR_MOD, HB_STUDY_T, LABELS_QUERY


class HypothesesBuilder:
    """ Main class to build n-ary hypotheses from CoDa Databank """
    def __init__(self, type_h: str, es_measure: str, mod_to_category: Union[dict, None] = None):
        """ Init main variables 
        - `type_h` is the type of hypothesis to extract, either `regular`,
        `numerical` or `categorical` 

        """

        # type_h_o = ['regular', 'numerical', 'categorical']
        type_h_o = ['regular', 'var_mod', 'study_mod']
        if type_h not in type_h_o:
            raise ValueError(f"Parameter `type_h` must be in {type_h_o}")
        self.type_h = type_h

        es_measure_o = ['d', 'r']
        if es_measure not in es_measure_o:
            raise ValueError(f"Parameter `es_measure` must be in {es_measure_o}")
        self.es_measure = es_measure

        self.type_h_to_query = {
            'regular': HB_REGULAR_T.replace("[es_measure]", es_measure),
            'var_mod': HB_REGULAR_T.replace("[es_measure]", es_measure),
            'study_mod': HB_STUDY_T.replace("[es_measure]", es_measure),
        }
        self.type_h_to_treat_vals = {
            'regular': TREATMENT_VALS_T_REGULAR,
            'var_mod': TREATMENT_VALS_T_VAR_MOD,
            'study_mod': TREATMENT_VALS_T_REGULAR,
        }
        self.type_h_to_make_blank_h = {
            'regular': self.make_final_iv,
            'var_mod': self.make_final_iv,
            'study_mod': self.make_final_iv,
        }
        self.type_h_to_iv_and_cats_cols = {
            'regular': ['iv', 'cat_t1', 'cat_t2', 'number_iv'],
            'var_mod': ['iv', 'cat_t1', 'cat_t2', 'number_iv',
                        'mod', 'mod_t1', 'mod_t2', 'number_mod'],
            'study_mod': ['iv', 'cat_t1', 'cat_t2', 'number_iv',
                          'mod', 'mod_val', 'number_mod'],
        }

        self.treat_prop_filter_out = [
            'http://www.w3.org/2000/01/rdf-schema#label',
            'https://data.cooperationdatabank.org/vocab/prop/meanContributionOrWithdrawalForCondition',
            'https://data.cooperationdatabank.org/vocab/prop/nCondition',
            'https://data.cooperationdatabank.org/vocab/prop/sDforCondition', 
            'https://data.cooperationdatabank.org/vocab/prop/proportionOfCooperationCondition', 
            'https://data.cooperationdatabank.org/vocab/prop/individualDifferenceLevel',
            'https://data.cooperationdatabank.org/vocab/prop/nbOfLevels']
        self.es_bins = [-np.inf, -0.5, -0.2, 0.2, 0.5, np.inf]
        self.es_labels = ['LargeMediumNegativeES', 'SmallNegativeES', 'NullFinding',
                          'SmallPositiveES', 'LargeMediumPositiveES']

        if mod_to_category:
            self.mod_filter_out = [mod for mod, val in mod_to_category.items() if val == "numerical"]
        else:
            self.mod_filter_out = []
        
        self.labels_query = LABELS_QUERY

    def get_observations(self, sparql_endpoint):
        """ Get observations to build the n-ary hypotheses """
        logger.info("Retrieving treatments")
        observations = run_query(
            query=self.type_h_to_query[self.type_h],
            sparql_endpoint=sparql_endpoint,
            headers=HEADERS_CSV)
        observations = observations.astype('object')

        logger.info("Retrieving info about treatments")
        t1_t2 = observations[['t1', 't2']].drop_duplicates()

        df_treat = None
        for _, row in tqdm(t1_t2.iterrows(), total=len(t1_t2)):
            curr_treat = run_query(
                query=self.type_h_to_treat_vals[self.type_h].replace("[iri1]", row.t1).replace("[iri2]", row.t2),
                sparql_endpoint=sparql_endpoint,
                headers=HEADERS_CSV)
            # Filter out some values: (1) same treatment values 
            # (2) certain properties (3) literal objects
            for (pred, col1, col2) in [("iv", "o1", "o2"), ("mod", "mod_t1", "mod_t2")]:
                if all(x in curr_treat.columns for x in (pred, col1, col2)):
                    curr_treat[col1] = curr_treat[col1].astype("string")
                    curr_treat[col2] = curr_treat[col2].astype("string")
                    curr_treat = curr_treat[(curr_treat[col1] != curr_treat[col2]) & \
                        (~curr_treat.iv.isin(self.treat_prop_filter_out))]
                    curr_treat = curr_treat[(curr_treat[col1].str.startswith('http')) & \
                        (curr_treat[col2].str.startswith('http'))]

            if isinstance(df_treat, pd.DataFrame):
                df_treat = pd.concat([df_treat, curr_treat])
            else:
                df_treat = curr_treat

        logger.info("Merging treatment info with original info + post-processing")
        observations = pd.merge(observations, df_treat, on=['t1', 't2'], how='left')
        observations = observations.dropna(subset=['cat_t1', 'cat_t2']) \
            .reset_index(drop=True)

        # Removing blank nodes with info
        for col in [x for x in observations.columns if x.startswith("mod")]:
            observations = observations[observations[col].notna()]
            observations = observations[~observations[col].str.contains(".well-known", na=False)]
        
        # Keeping only categorical moderators for now
        if "mod" in observations.columns:
            observations = observations[~observations["mod"].isin(self.mod_filter_out)]

        return observations

    @staticmethod
    def get_num_hypothesis_three(obs, row, i, cols, ivs_and_cats):
        """ Get num of hypothesis for a set of cols
        cols: [pred, val1, val2]
        works both for moderators and regular variables 
        
        cols:
        - ['iv', 'cat_t1', 'cat_t2']
        - ['mod', 'mod_t1', 'mod_t2'] """
        pred, val1, val2 = row[cols[0]], row[cols[1]], row[cols[2]]
        update_ivs_and_cats = False

        filtered = ivs_and_cats[
            (ivs_and_cats[cols[0]] == pred) & \
                (ivs_and_cats[cols[1]] == val1) & \
                    (ivs_and_cats[cols[2]] == val2)
        ]
        filtered_reverse = ivs_and_cats[
            (ivs_and_cats[cols[0]] == pred) & \
                (ivs_and_cats[cols[1]] == val2) & \
                    (ivs_and_cats[cols[2]] == val1)
        ]
        f_shape, fr_shape = filtered.shape[0], filtered_reverse.shape[0]

        if (f_shape == 0) and (fr_shape == 0):

            try:
                num = ivs_and_cats[cols[0]].value_counts()[pred]+1
            except:
                num = 1
            update_ivs_and_cats = True

        elif f_shape > 0:
            num = ivs_and_cats.loc[(ivs_and_cats[cols[0]] == pred) & \
                                   (ivs_and_cats[cols[1]] == val1) & \
                                   (ivs_and_cats[cols[2]] == val2)][cols[3]].values[0]

        else:  # fr_shape > 0
            obs.at[i, cols[1]] = val2
            obs.at[i, cols[2]] = val1
            if cols[0] == 'iv':
                obs.at[i, 'ES'] = row['ES'] * -1
                obs.at[i, 'ESUpper'] = row['ESUpper'] * -1
                obs.at[i, 'ESLower'] = row['ESLower'] * -1

            num = ivs_and_cats.loc[(ivs_and_cats[cols[0]] == pred) & \
                                   (ivs_and_cats[cols[1]] == val2) & \
                                   (ivs_and_cats[cols[2]] == val1)][cols[3]].values[0]
        
        return num, obs, ivs_and_cats, update_ivs_and_cats

    @staticmethod
    def get_num_hypothesis_two(obs, row, i, cols, ivs_and_cats):
        """ Get num of hypothesis for a set of cols of length 2
        cols: [pred, val]
        
        cols:
        - ['mod', 'mod_val', 'number_mod'] """
        pred, val = row[cols[0]], row[cols[1]]
        update_ivs_and_cats = False

        filtered = ivs_and_cats[
            (ivs_and_cats[cols[0]] == pred) & \
                (ivs_and_cats[cols[1]] == val)
        ]
        f_shape = filtered.shape[0]


        if f_shape == 0:

            try:
                num = ivs_and_cats[cols[0]].value_counts()[pred]+1
            except:
                num = 1
            update_ivs_and_cats = True

        else:
            num = ivs_and_cats.loc[(ivs_and_cats[cols[0]] == pred) & \
                                   (ivs_and_cats[cols[1]] == val)][cols[2]].values[0]

        return num, obs, ivs_and_cats, update_ivs_and_cats

    def make_final_iv(self, obs):
        """ Count hypothesis number """
        ivs_and_cats = pd.DataFrame(columns=self.type_h_to_iv_and_cats_cols[self.type_h])

        for i, row in tqdm(obs.iterrows(), total=len(obs)):
            if self.type_h == "regular":
                num, obs, ivs_and_cats, update_reg = self.get_num_hypothesis_three(
                    obs=obs, row=row, i=i, cols=['iv', 'cat_t1', 'cat_t2', 'number_iv'],
                    ivs_and_cats=ivs_and_cats)
                if update_reg:
                    df_row = pd.DataFrame([[row['iv'], row['cat_t1'], row['cat_t2'], num]],
                                            columns=self.type_h_to_iv_and_cats_cols[self.type_h])
                    ivs_and_cats = pd.concat([ivs_and_cats, df_row])
                obs.at[i, 'iv_new'] = row['iv'] + str('_H') + str(num)

            if self.type_h == "var_mod":
                num_1, obs, ivs_and_cats, update_reg = self.get_num_hypothesis_three(
                    obs=obs, row=row, i=i, cols=['iv', 'cat_t1', 'cat_t2', 'number_iv'],
                    ivs_and_cats=ivs_and_cats)
                num_2, obs, ivs_and_cats, update_var_mod = self.get_num_hypothesis_three(
                    obs=obs, row=row, i=i, cols=['mod', 'mod_t1', 'mod_t2', 'number_mod'],
                    ivs_and_cats=ivs_and_cats)
                if (update_reg or update_var_mod):
                    df_row = pd.DataFrame([[
                        row['iv'], row['cat_t1'], row['cat_t2'], num_1,
                        row['mod'], row['mod_t1'], row['mod_t2'], num_2]],
                        columns=self.type_h_to_iv_and_cats_cols[self.type_h])
                    ivs_and_cats = pd.concat([ivs_and_cats, df_row])
                obs.at[i, 'iv_new'] = row['iv'] + "_" + row["mod"] + str('_H') + str(num_1) + "_" + str(num_2)
            
            if self.type_h == "study_mod":
                num_1, obs, ivs_and_cats, update_reg = self.get_num_hypothesis_three(
                    obs=obs, row=row, i=i, cols=['iv', 'cat_t1', 'cat_t2', 'number_iv'],
                    ivs_and_cats=ivs_and_cats)
                num_2, obs, ivs_and_cats, update_var_mod = self.get_num_hypothesis_two(
                    obs=obs, row=row, i=i, cols=['mod', 'mod_val', 'number_mod'],
                    ivs_and_cats=ivs_and_cats)
                if (update_reg or update_var_mod):
                    df_row = pd.DataFrame([[
                        row['iv'], row['cat_t1'], row['cat_t2'], num_1,
                        row['mod'], row['mod_val'], num_2]],
                        columns=self.type_h_to_iv_and_cats_cols[self.type_h])
                    ivs_and_cats = pd.concat([ivs_and_cats, df_row])
                obs.at[i, 'iv_new'] = row['iv'] + "_" + row["mod"] + str('_H') + str(num_1) + "_" + str(num_2)

        return obs

    def bin_effect_size(self, obs):
        """ Categorizing effect sizes """
        categories = pd.cut(obs['ES'], self.es_bins, labels=self.es_labels)
        obs['ESType'] = categories
        return obs

    def add_labels(self, obs, sparql_endpoint):
        """ Adding labels """
        labels = run_query(
            query=self.labels_query,
            sparql_endpoint=sparql_endpoint,
            headers=HEADERS_CSV
        ).set_index("n")['nl'].to_dict()
        for col in ["iv", "cat_t1", "cat_t2", "mod", "mod_t1", "mod_t2", "mod_val"]:
            if col in obs.columns:
                obs[f"{col}_label"] = obs[col].apply(lambda x: labels.get(x, "na"))
        return obs

    def __call__(self, sparql_endpoint, observations: Union[str, None] = None,
                 triples: Union[str, None] = None):
        if observations:
            observations = pd.read_csv(observations, index_col=0)
        else:
            observations = self.get_observations(sparql_endpoint=sparql_endpoint)
            logger.info("Adding hypotheses as blank nodes")
            observations = self.type_h_to_make_blank_h[self.type_h](observations)
            logger.info("Categorizing effect sizes")
            observations = self.bin_effect_size(obs=observations)
            logger.info("Adding labels")
            observations = self.add_labels(obs=observations, sparql_endpoint=sparql_endpoint)

        return observations


if __name__ == '__main__':
    SPARQL_ENDPOINT = "http://localhost:7200/repositories/coda"
    # TRIPLES = "./triples_32072.csv"
    OBSERVATIONS, TRIPLES = None, None
    OBSERVATIONS = "obs_var_mod.csv"
    HB = HypothesesBuilder(type_h='regular', es_measure='d')
    # HB(sparql_endpoint=SPARQL_ENDPOINT, triples=TRIPLES)
    OBS_VAR_MOD = HB(sparql_endpoint=SPARQL_ENDPOINT)
    OBS_VAR_MOD.to_csv("obs_study_mod.csv")

    # OBS = HB.get_observations(sparql_endpoint=SPARQL_ENDPOINT)
