# -*- coding: utf-8 -*-
"""
From a trained KG embedding model, retrieve and save embeddings
"""
import gzip
from tqdm import tqdm
from typing import List
import pandas as pd
import numpy as np
import torch

def read_gzip(path):
    """ Read csv encoded as gzip """
    with gzip.open(path, 'rt') as f:
        return pd.read_csv(f, sep="\t", names=['id', 'label'], header=0)

class HypothesesEmbedder:
    """
    Takes
    - a KG model
    - mappings
    - data
    as input and returns a vector
    """
    def __init__(self, columns, type_columns, target, model_path, entity_to_id_path, relation_to_id_path, classes_to_id):
        """ Columns to embed and their types (`entity` or `relation`) """
        self.columns = columns
        self.type_columns = type_columns
        self.target = target
        self.model = torch.load(model_path)
        entity_df = read_gzip(path=entity_to_id_path)
        self.entity_to_id = pd.Series(entity_df["id"].values, index=entity_df.label).to_dict()
        relation_df = read_gzip(path=relation_to_id_path)
        self.relation_to_id = pd.Series(relation_df["id"].values, index=relation_df.label).to_dict()
        self.classes_to_id = classes_to_id

        self.embedding_dim = self.model.entity_representations[0]().shape[1]
    
    def get_embedding(self, index, val):
        """ Retrieve embedding """
        if self.type_columns[index] == "entity":
            return self.model.entity_representations[0]().detach()[self.entity_to_id[val],:].cpu().numpy()
        # self.type_columns[index] == "relation":
        return self.model.entity_representations[0]().detach()[self.entity_to_id[val],:].cpu().numpy()
    
    def __call__(self, data):
        """ Concatenate and embed data """
        output_x = np.empty((0, self.embedding_dim*len(self.columns)))
        output_y = np.zeros((1, data.shape[0]))
        
        for index, row in tqdm(data.iterrows(), total=data.shape[0]):
            vals = [row[col] for col in self.columns]
            curr_emb = np.concatenate([self.get_embedding(index=i, val=val) for i, val in enumerate(vals)], axis=None)
            output_x = np.concatenate((output_x, curr_emb.reshape(1, -1)), axis=0)
            output_y[0, index] = self.classes_to_id[row[self.target]]
        
        return output_x, output_y



if __name__ == '__main__':
    COLUMNS = ["iv", "cat_t1", "iv", "cat_t2"]
    TYPE_COLUMNS = ["entity", "entity", "entity", "entity"]
    TARGET = "ESType"
    MODEL_PATH = "./kg_model_test/trained_model.pkl"
    ENTITY_TO_ID_PATH = "./kg_model_test/training_triples/entity_to_id.tsv.gz"
    RELATION_TO_ID_PATH = "./kg_model_test/training_triples/relation_to_id.tsv.gz"
    CLASSES_TO_ID = {
        'LargeMediumNegativeES': 0, 'SmallNegativeES': 1, 'NullFinding': 2,
        'SmallPositiveES': 3, 'LargeMediumPositiveES': 4
    }
    HE = HypothesesEmbedder(
        columns=COLUMNS, type_columns=TYPE_COLUMNS, target=TARGET, model_path=MODEL_PATH,
        entity_to_id_path=ENTITY_TO_ID_PATH, relation_to_id_path=RELATION_TO_ID_PATH,
        classes_to_id=CLASSES_TO_ID
    )

    DATA = pd.read_csv("./data/hypotheses/entry/h_regular_es_d.csv", index_col=0)[:100]
    X, Y = HE(data=DATA)
    np.save("X.npy", X)
    np.save("Y.npy", Y)