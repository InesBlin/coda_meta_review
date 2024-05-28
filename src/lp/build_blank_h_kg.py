# -*- coding: utf-8 -*-
"""
Construct KG for blank node hypothesis methods
"""
import pandas as pd
from tqdm import tqdm
from src.lp.sparql_queries import EFFECT_CONSTRUCT_T

def rdflib_to_pd(graph):
    """ Rdflib graph to pandas df with columns ["subject", "predicate", "object"] """
    df = pd.DataFrame(columns=['subject', 'predicate', 'object'])
    for subj, pred, obj in graph:
        df.loc[df.shape[0]] = [str(subj), str(pred), str(obj)]
    return df

class BlankHypothesesKGBuilder:
    """ Main class to build the KGs for automated generation via LP """

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


if __name__ == "__main__":
    BHKGB = BlankHypothesesKGBuilder()

""" 
ORIG: ?study cp:reportsEffect <[obs]> . 
NEW: <row.study> cp:reportsEffect <row.obs>
(
    <row.study> <row.mod> <row.mod_val> 
)

ORIG: <[obs]> cp:dependentVariable ?dependentVariable .
NEW: <row.obs> cp:ESType class:<row.dependent> 

ORIG: <[obs]> cp:ESType  class:[es_type] . 
NEW: <row.obs> cp:ESType class:<row.ESType>

ORIG: <[obs]> cp:treatment ?t1
NEW: <row.obs> cp:treatment <row.t1>

ORIG: <[obs]> cp:treatment ?t2
NEW: <row.obs> cp:treatment <row.t2>

ORIG: <[iv]> rdfs:subPropertyOf ?superProperty . 
NEW: 
<row.iv> rdfs:subPropertyOf <row.giv_prop>
(
    <row.mod> rdfs:subPropertyOf <row.giv_prop>  
)

ORIG: 
[line_t1]
[line_t2]
NEW: 
cf. l21-26
+ 
<row.t1> <row.mod> <row.mod_t1>
<row.t2> <row.mod> <row.mod_t2>

ORIG: [cp_effect]
NEW: cf. l28-32

ORIG:
<[iv]> rdfs:range ?class . 
?class rdfs:subClassOf ?superClass . 
?superClass rdfs:subClassOf class:IndependentVariable .
NEW:
<row.iv> rdfs:range <row.range_class_iv> 
<row.class_iv> rdfs:subClassOf <row.range_superclass_iv> 
<row.range_superclass_iv> rdfs:subClassOf class:IndependentVariable
(
    <row.mod> rdfs:range <row.range_class_mod> 
    <row.range_class_mod> rdfs:subClassOf <row.range_superclass_mod> 
    <row.range_superclass_mod> rdfs:subClassOf class:IndependentVariable
)

ORIG: <[iv_h]> rdfs:subPropertyOf  <[iv]> . 
NEW: <row.iv_new> rdfs:subPropertyOf <row.iv>

NEW:
<row.iv_new> cdp:sivv1 <row.cat_t1>
<row.iv_new> cdp:sivv2 <row.cat_t2>
(
    <row.iv_new> cdp:mod1 <row.mod_t1>
    <row.iv_new> cdp:mod2 <row.mod_t2>
    <row.iv_new> cdp:mod <row.mod>
)

"""