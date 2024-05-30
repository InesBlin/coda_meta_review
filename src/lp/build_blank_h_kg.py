# -*- coding: utf-8 -*-
"""
Construct KG for blank node hypothesis methods
"""
import math
from typing import Union
import pandas as pd
from tqdm import tqdm
from kglab.helpers.variables import STR_RDF, STR_RDFS
from src.lp.sparql_queries import EFFECT_CONSTRUCT_T

def rdflib_to_pd(graph):
    """ Rdflib graph to pandas df with columns ["subject", "predicate", "object"] """
    df = pd.DataFrame(columns=['subject', 'predicate', 'object'])
    for subj, pred, obj in graph:
        df.loc[df.shape[0]] = [str(subj), str(pred), str(obj)]
    return df


def type_of_effect(row):
    """ Categorize effect based on its signifiance """
    if math.isnan(row.ESLower) or math.isnan(row.ESUpper):
        if row.ES > -0.2 and row.ES < 0.2:
            return 'noEffect'
        return 'positive' if row.ES >= 0.2 else 'negative'
    if row.ESLower <= 0 <= row.ESUpper:
        return 'noEffect'
    return 'positive'  if float(row.ES) > 0 else 'negative'


class BlankHypothesesKGBuilder:
    """ Main class to build the KGs for automated generation via LP """
    def __init__(self):
        self.prefixes = {
            "cp": "https://data.cooperationdatabank.org/vocab/prop/",
            "rdf": STR_RDF,
            "rdfs": STR_RDFS,
            "id": "https://data.cooperationdatabank.org/id/",
            "class": "https://data.cooperationdatabank.org/vocab/class/"
        }
        self.effect_to_prop = {
            'positive': f"{self.prefixes['cp']}hasPositiveEffectOn",
            'negative': f"{self.prefixes['cp']}hasNegativeEffectOn",
            'noEffect': f"{self.prefixes['cp']}hasNoEffectOn"
        }
        self.triples_cols = ["subject", "predicate", "object"]

        self.triples_info = [
            # Original data in the graph
            ("study", f"{self.prefixes['cp']}reportsEffect", "obs", {}),
            ("obs", f"{self.prefixes['cp']}ESType", "dependent", {}),
            ("obs", f"{self.prefixes['cp']}ESType", "ESType", {"ESType": self.prefixes['class']}),
            ("obs", f"{self.prefixes['cp']}treatment", "t1", {}),
            ("obs", f"{self.prefixes['cp']}treatment", "t2", {}),
            ("iv", f"{self.prefixes['rdfs']}subPropertyOf", "giv_prop", {}),
            ("mod", f"{self.prefixes['rdfs']}subPropertyOf", "giv_prop", {}),
            ("t1", "iv", "cat_t1", {}),
            ("t2", "iv", "cat_t2", {}),
            ("t1", "mod", "mod_t1", {}),
            ("t2", "mod", "mod_t2", {}),
            ("study", "mod", "mod_val", {}),
            ("iv", f"{self.prefixes['rdfs']}range", "range_class_iv", {}),
            ("class_iv", f"{self.prefixes['rdfs']}subClassOf", "range_superclass_iv", {}),
            ("mod", f"{self.prefixes['rdfs']}range", "range_class_mod", {}),
            ("range_class_mod", f"{self.prefixes['rdfs']}subClassOf", "range_superclass_mod", {}),

            # Newly added data with hypotheses
            ("iv_new", f"{self.prefixes['rdfs']}subPropertyOf", "iv", {}),
            ("iv_new", f"{self.prefixes['cp']}sivv1", "cat_t1", {}),
            ("iv_new", f"{self.prefixes['cp']}sivv2", "cat_t2", {}),
            ("iv_new", f"{self.prefixes['cp']}mod1", "mod_t1", {}),
            ("iv_new", f"{self.prefixes['cp']}mod2", "mod_t2", {}),
            ("iv_new", f"{self.prefixes['cp']}mod", "mod", {}),
        ]
        self.cols_range_superclass = ["range_superclass_iv", "range_superclass_mod"]

    
    def build_triples(self, data: pd.DataFrame, subj: str, pred: str, obj: str, **prefixes):
        """ Build triple data from an original dataframe
        - `subject` and `object` are columns in `data`
        - `predicate` is the predicate value in the triple data
        Overall: removes duplicates and construct triple data """
        if (subj in data.columns) and (obj in data.columns):
            if pred in data.columns:
                df = data[[subj, pred, obj]].drop_duplicates()
                df = df[(~df[subj].isna()) & (~df[obj].isna()) & (~df[pred].isna())]
                for k, val in prefixes.items():
                    df[k] = val + df[k]
                values = [[row[subj], row[pred], row[obj]] for _, row in df.iterrows()]
            else:
                df = data[[subj, obj]].drop_duplicates()
                df = df[(~df[subj].isna()) & (~df[obj].isna())]
                for k, val in prefixes.items():
                    df[k] = val + df[k]
                values = [[row[subj], pred, row[obj]] for _, row in df.iterrows()]
        else:
            values = []
        return pd.DataFrame(values, columns=self.triples_cols)
    
    def add_effect(self, data: pd.DataFrame):
        """ Characterise effect size based on confidence intervals """
        columns = ["iv_new", "ES", "ESLower", "ESUpper", "dependent"]
        df = data[columns].drop_duplicates()
        tqdm.pandas()
        df["effect"] = df.progress_apply(type_of_effect, axis=1)
        values = [(row.iv_new, self.effect_to_prop[row.effect], row.dependent) for _, row in df.iterrows()]
        return pd.DataFrame(values, columns=self.triples_cols)
    
    def add_range_superclass(self, data, col_name):
        """ Add ontology info about superclass """
        if col_name in data.columns:
            values = [(x, f"{self.prefixes['rdfs']}subClassOf", f"{self.prefixes['class']}IndependentVariable") \
                for x in data[col_name].unique()]
        else:
            values = []
        return pd.DataFrame(values, columns=self.triples_cols)
        

    def instantiate_effect_query(self, row):
        """ [DEPRECATED] not used
        Replace templated content from  EFFECT_CONSTRUCT_T with real values """
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
        """ [DEPRECATED] not used
        Construct KG with n-ary hypotheses """
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

    def __call__(self, data, vocab: Union[pd.DataFrame, None] = None):
        """ Build KG in .csv format from `data` 
        Return two KGs, easiest for future link prediction tasks:
        - One KG that will be randomly split across training/validation/testing
        - One KG that will need further refinement for training/validation/testing
            (distributed evenly per generic independent variable) """
        output = pd.DataFrame(columns=self.triples_cols)
        for subj, pred, obj, prefixes in tqdm(self.triples_info):
            curr_df = self.build_triples(data=data, subj=subj, pred=pred, obj=obj, **prefixes)
            output = pd.concat([output, curr_df])
        
        for col_name in tqdm(self.cols_range_superclass):
            curr_df = self.add_range_superclass(data=data, col_name=col_name)
            output = pd.concat([output, curr_df])

        # Add vocab if applicable
        if isinstance(vocab, pd.DataFrame) and all(x in vocab.columns for x in self.triples_cols):
            df_vocab = pd.DataFrame([row[col] for col in self.triples_cols] for _, row in vocab.iterrows())
            output = pd.concat([output, df_vocab])
        
        # Add effect
        df_effect = self.add_effect(data=data)

        return output.drop_duplicates(), df_effect

if __name__ == "__main__":
    BHKGB = BlankHypothesesKGBuilder()
    DATA = pd.read_csv("./data/hypotheses/entry/h_study_mod_es_d.csv", index_col=0)
    VOCAB = pd.read_csv("./data/vocab.csv", index_col=0)
    RES, _ = BHKGB(data=DATA, vocab=VOCAB)
    RES.to_csv("res.csv")

