# -*- coding: utf-8 -*-
"""
KG Embedder
"""
from typing import List
import pandas as pd
from pykeen.triples import TriplesFactory

class KGEmbedder:
    """ 
    - Embedding KG nodes
    - Possibility to initialise from pre-trained embeddings
    """
    def __init__(self, data_path: str, spo_cols: List[str]):
        """
        - data_path: .csv path with data
        - spo_cols: names of subject, predicate, object cols
        """
        dc = 'http://purl.org/dc/terms/'
        cdp = 'https://data.cooperationdatabank.org/vocab/prop/'
        self.filter_out_pred = [
            'http://www.w3.org/2000/01/rdf-schema#label',
            dc + 'description',
            dc + 'title',
            cdp + 'descriptionIV',
            'http://www.w3.org/2002/07/owl#sameAs',
            cdp + "otherVariables"
        ]
        df = pd.read_csv(data_path)
        df = df[~df[spo_cols[1]].isin(self.filter_out_pred)]
        for col in spo_cols:
            df = df[~df[col].isna()]
        self.sh = TriplesFactory.from_labeled_triples(
            df[spo_cols].values,
            create_inverse_triples=True)


if __name__ == '__main__':
    DATA_PATH = "./data/coda_kg.csv"
    SPO_COLS = ['subject', 'predicate', 'object']
    KG_EMB = KGEmbedder(data_path=DATA_PATH, spo_cols=SPO_COLS)
    print(KG_EMB.sh.entity_to_id)