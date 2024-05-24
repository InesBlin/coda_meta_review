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
from src.lp.sparql_queries import HB_REGULAR_T, TREATMENT_VALS_T_REGULAR, EFFECT_CONSTRUCT_T, \
    TREATMENT_VALS_T_VAR_MOD, HB_STUDY_T

def type_of_effect(effect_size, lower, upper):
    """ Categorize effect based on its signifiance """
    if math.isnan(lower) or math.isnan(upper):
        if effect_size > -0.2 and effect_size < 0.2:
            return 'noEffect'
        return 'positive' if effect_size >= 0.2 else 'negative'
    if lower <= 0 <= upper:
        return 'noEffect'
    return 'positive'  if float(effect_size) > 0 else 'negative'

def rdflib_to_pd(graph):
    """ Rdflib graph to pandas df with columns ["subject", "predicate", "object"] """
    df = pd.DataFrame(columns=['subject', 'predicate', 'object'])
    for subj, pred, obj in graph:
        df.loc[df.shape[0]] = [str(subj), str(pred), str(obj)]
    return df


class HypothesesBuilder:
    """ Main class to build n-ary hypotheses from CoDa Databank """
    def __init__(self, type_h: str, es_measure: str):
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
        df_treat = pd.DataFrame(columns=["t1", "t2", "p", "o1", "o2"])
        for _, row in tqdm(t1_t2.iterrows(), total=len(t1_t2)):
            curr_treat = run_query(
                query=self.type_h_to_treat_vals[self.type_h].replace("[iri1]", row.t1).replace("[iri2]", row.t2),
                sparql_endpoint=sparql_endpoint,
                headers=HEADERS_CSV)
            # Filter out some values: (1) same treatment values 
            # (2) certain properties (3) literal objects
            for (pred, col1, col2) in [("p", "o1", "o2"), ("mod", "mod_t1", "mod_t2")]:
                if all(x in curr_treat.columns for x in (pred, col1, col2)):
                    curr_treat[col1] = curr_treat[col1].astype("string")
                    curr_treat[col2] = curr_treat[col2].astype("string")
                    curr_treat = curr_treat[(curr_treat[col1] != curr_treat[col2]) & \
                        (~curr_treat.p.isin(self.treat_prop_filter_out))]
                    curr_treat = curr_treat[(curr_treat[col1].str.startswith('http')) & \
                        (curr_treat[col2].str.startswith('http'))]
            df_treat = pd.concat([df_treat, curr_treat])

        logger.info("Merging treatment info with original info + post-processing")
        observations = pd.merge(observations, df_treat, on=['t1', 't2'], how='left')
        observations = observations.rename(columns={
            "p": "iv", "o1": "cat_t1", "o2": "cat_t2"})
        observations = observations.dropna(subset=['cat_t1', 'cat_t2']) \
            .reset_index(drop=True)

        return observations

    @staticmethod
    def make_final_iv_regular(obs):
        """ Count hypothesis number """
        sets, lists = [], []
        ivs_and_cats = pd.DataFrame(columns=['iv', 'cat_t1', 'cat_t2', 'number'])

        for i, row in tqdm(obs.iterrows(), total=len(obs)):
            iv = row['iv']
            cat_t1, cat_t2 = row['cat_t1'], row['cat_t2']
            es, es_up, es_lo = row['ES'], row['ESUpper'], row['ESLower']

            check_list = [iv, cat_t1, cat_t2]
            check_set = set(check_list)

            if check_set not in sets and check_list not in lists:
                lists.append(check_list)
                sets.append(check_set)

                try:
                    num = ivs_and_cats['iv'].value_counts()[iv]+1
                except:
                    num = 1
                df_row = pd.DataFrame([[iv, cat_t1, cat_t2, num]],
                                      columns = ['iv', 'cat_t1', 'cat_t2', 'number'])
                ivs_and_cats = pd.concat([ivs_and_cats, df_row])

                obs.at[i, 'iv_new'] = iv + str('_H') + str(num) 

            elif check_set in sets and check_list in lists: 
                index = ivs_and_cats.loc[(ivs_and_cats['iv'] == iv) & 
                                        (ivs_and_cats['cat_t1'] == cat_t1) &
                                        (ivs_and_cats['cat_t2'] == cat_t2)]['number']
                obs.at[i, 'iv_new'] = iv + str('_H') + str(index.values[0])  

            else:
                obs.at[i, 'cat_t1'] = cat_t2
                obs.at[i, 'cat_t2'] = cat_t1
                obs.at[i, 'ES'] = es * -1
                obs.at[i, 'ESUpper'] = es_up * -1
                obs.at[i, 'ESLower'] = es_lo * -1

                index = ivs_and_cats.loc[(ivs_and_cats['iv'] == iv) & 
                                         (ivs_and_cats['cat_t1'] == cat_t2) &
                                         (ivs_and_cats['cat_t2'] == cat_t1)]['number']
                obs.at[i, 'iv_new'] = iv + str('_H') + str(index.values[0])

        return obs
    
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

    def instantiate_effect_query(self, row):
        """ Replace templated content from  EFFECT_CONSTRUCT_T with real values """
        if checkers.is_url(row.cat_t1) and checkers.is_url(row.cat_t2): 
            line_t1 = '?t1 <' + row.iv + '> <' + row.cat_t1 + '> . '
            line_t2 = '?t2 <' + row.iv + '> <' + row.cat_t2 + '> . '
        else:         
            line_t1 = '?t1 <' + row.iv + '> "' + str(row.cat_t1) + '" .'
            line_t2 = '?t2 <' + row.iv + '> "' + str(row.cat_t2) + '" .'

        effect = type_of_effect(row.ES, row.ESLower, row.ESUpper)
        if effect == 'positive':
            effect_prop = 'hasPositiveEffectOn'
        elif effect == 'negative':
            effect_prop = 'hasNegativeEffectOn'
        else:
            effect_prop = 'hasNoEffectOn'
        cp_effect = '<' + row.iv_new + '>' + ' cp:' + effect_prop + ' ?dependentVariable .'

        return EFFECT_CONSTRUCT_T.replace("[cp_effect]", cp_effect) \
            .replace("[iv_h]", row.iv_new) \
                .replace("[es_measure]", self.es_measure) \
                    .replace("[obs]", row.obs) \
                        .replace("[line_t1]", line_t1) \
                            .replace("[line_t2]", line_t2) \
                                .replace("[iv]", row.iv) \
                                    .replace("[es_type]", row.ESType)

    def construct_effect_kg(self, obs, sparql_endpoint):
        """ Construct KG with n-ary hypotheses """
        triples = pd.DataFrame(columns=['subject', 'predicate', 'object'])
        for _, row in tqdm(obs.iterrows(), total=len(obs)):
            query = self.instantiate_effect_query(row=row)
            graph = Graph()
            response = run_query(query=query,
                                    sparql_endpoint=sparql_endpoint,
                                    headers=HEADERS_RDF_XML)
            graph.parse(data=response.text, format='xml')
            triples = pd.concat([triples, rdflib_to_pd(graph=graph)])
        return triples

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

        return observations

        # if triples:
        #     triples = pd.read_csv(triples, index_col=0)
        # else:
        #     logger.info("Constructing KG")
        #     triples = self.construct_effect_kg(obs=observations.sample(10000),
        #                                        sparql_endpoint=sparql_endpoint)
        #     triples.to_csv(f"triples_{triples.shape[0]}.csv")

        # return triples


if __name__ == '__main__':
    SPARQL_ENDPOINT = "http://localhost:7200/repositories/coda"
    # TRIPLES = "./triples_32072.csv"
    OBSERVATIONS, TRIPLES = None, None
    OBSERVATIONS = "obs_var_mod.csv"
    HB = HypothesesBuilder(type_h='study_mod', es_measure='d')
    # HB(sparql_endpoint=SPARQL_ENDPOINT, triples=TRIPLES)
    OBS_VAR_MOD = HB(sparql_endpoint=SPARQL_ENDPOINT)
    OBS_VAR_MOD.to_csv("obs_study_mod.csv")

    # OBS = HB.get_observations(sparql_endpoint=SPARQL_ENDPOINT)
