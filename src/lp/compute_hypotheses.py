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
from src.lp.sparql_queries import HB_REGULAR_T, TREATMENT_VALS_T, EFFECT_CONSTRUCT_T

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

        type_h_o = ['regular', 'numerical', 'categorical']
        if type_h not in type_h_o:
            raise ValueError(f"Parameter `type_h` must be in {type_h_o}")
        self.type_h = type_h

        es_measure_o = ['d', 'r']
        if es_measure not in es_measure_o:
            raise ValueError(f"Parameter `es_measure` must be in {es_measure_o}")
        self.es_measure = es_measure

        self.type_h_to_query = {
            'regular': HB_REGULAR_T.replace("[es_measure]", es_measure)
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
                query=TREATMENT_VALS_T.replace("[iri1]", row.t1).replace("[iri2]", row.t2),
                sparql_endpoint=sparql_endpoint,
                headers=HEADERS_CSV)
            # Filter out some values: (1) same treatment values 
            # (2) certain properties (3) literal objects
            curr_treat = curr_treat[(curr_treat.o1 != curr_treat.o2) & \
                (~curr_treat.p.isin(self.treat_prop_filter_out))]
            curr_treat = curr_treat[(curr_treat.o1.str.startswith('http')) & \
                (curr_treat.o2.str.startswith('http'))]
            df_treat = pd.concat([df_treat, curr_treat])
        df_treat.to_csv("df_treat.csv")

        logger.info("Merging treatment info with original info + post-processing")
        observations = pd.merge(observations, df_treat, on=['t1', 't2'], how='left')
        observations = observations.rename(columns={
            "p": "independentProperties", "o1": "cat_t1", "o2": "cat_t2"})
        observations = observations.dropna(subset=['cat_t1', 'cat_t2']) \
            .reset_index(drop=True)

        return observations

    @staticmethod
    def make_final_iv(obs):
        """ Count hypothesis number """
        sets, lists = [], []
        ivs_and_cats = pd.DataFrame(columns=['iv', 'cat_t1', 'cat_t2', 'number'])

        for i, row in tqdm(obs.iterrows(), total=len(obs)):
            iv = row['independentProperties']
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

    def bin_effect_size(self, obs):
        """ Categorizing effect sizes """
        categories = pd.cut(obs['ES'], self.es_bins, labels=self.es_labels)
        obs['ESType'] = categories
        return obs

    def instantiate_effect_query(self, row):
        """ Replace templated content from  EFFECT_CONSTRUCT_T with real values """
        if checkers.is_url(row.cat_t1) and checkers.is_url(row.cat_t2): 
            line_t1 = '?t1 <' + row.independentProperties + '> <' + row.cat_t1 + '> . '
            line_t2 = '?t2 <' + row.independentProperties + '> <' + row.cat_t2 + '> . '
        else:         
            line_t1 = '?t1 <' + row.independentProperties + '> "' + str(row.cat_t1) + '" .'
            line_t2 = '?t2 <' + row.independentProperties + '> "' + str(row.cat_t2) + '" .'

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
                                .replace("[iv]", row.independentProperties) \
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
            observations = self.make_final_iv(observations)
            logger.info("Categorizing effect sizes")
            observations = self.bin_effect_size(obs=observations)
            observations.to_csv("observations_iv.csv")
            observations[["cat_t1", "cat_t2", "iv_new"]].drop_duplicates().to_csv("hypotheses_readable.csv")

        print("observations: ", observations.shape)

        if triples:
            triples = pd.read_csv(triples, index_col=0)
        else:
            logger.info("Constructing KG")
            triples = self.construct_effect_kg(obs=observations.sample(10000),
                                               sparql_endpoint=sparql_endpoint)
            triples.to_csv(f"triples_{triples.shape[0]}.csv")

        return triples


if __name__ == '__main__':
    SPARQL_ENDPOINT = "http://localhost:7200/repositories/coda"
    # OBSERVATIONS = "observations_iv.csv"
    # TRIPLES = "./triples_32072.csv"
    OBSERVATIONS, TRIPLES = None, None
    HB = HypothesesBuilder(type_h='regular', es_measure='d')
    HB(sparql_endpoint=SPARQL_ENDPOINT, observations=OBSERVATIONS, triples=TRIPLES)
