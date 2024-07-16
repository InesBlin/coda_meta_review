# -*- coding: utf-8 -*-
"""
Save embeddings classification
"""
import os
import click
import pandas as pd
import numpy as np
from loguru import logger
from src.lp.embed_hypotheses import HypothesesEmbedder

ES_MEASURES = ["d", "r"]
LABELS = ["regular", "var_mod", "study_mod"]
COLUMNS = [
    ["giv_prop", "iv", "cat_t1", "iv", "cat_t2", "dependent"],
    ["giv_prop", "iv", "cat_t1", "iv", "cat_t2", "mod", "mod_t1", "mod_t2", "dependent"],
    ["giv_prop", "iv", "cat_t1", "iv", "cat_t2", "mod", "mod_val", "dependent"]
]
TYPE_COLUMNS = [["entity" for y in x] for x in COLUMNS]
TARGET = "effect"
MODEL_PATH = "./models/coda_ontology/trained_model.pkl"
ENTITY_TO_ID_PATH = "./models/coda_ontology/training_triples/entity_to_id.tsv.gz"
RELATION_TO_ID_PATH = "./models/coda_ontology/training_triples/relation_to_id.tsv.gz"
CLASSES_TO_ID = {
    'negative': 0, 'noEffect': 1, 'positive': 2
}

@click.command()
@click.argument("save_folder")
def main(save_folder):
    """ Retrieving data for all hypothesis type and es measures """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for i, cols in enumerate(COLUMNS):
        he = HypothesesEmbedder(
            columns=cols, type_columns=TYPE_COLUMNS[i], target=TARGET, model_path=MODEL_PATH,
            entity_to_id_path=ENTITY_TO_ID_PATH, relation_to_id_path=RELATION_TO_ID_PATH,
            classes_to_id=CLASSES_TO_ID
        )
        for es in ES_MEASURES:
            logger.info(f"Fetching embeddings for hypothesis `{LABELS[i]}` with effect size measure `{es}`")
            save_path = os.path.join(save_folder, f"h_{LABELS[i]}_es_{es}_x.npy")
            if not os.path.exists(save_path):
                data = pd.read_csv(f"./data/hypotheses/classification/h_{LABELS[i]}_es_{es}.csv", index_col=0).reset_index(drop=True)
                output_x, output_y = he(data=data)
                np.save(save_path, output_x)
                np.save(save_path.replace("_x", "_y"), output_y)


if __name__ == '__main__':
    # python experiments/hp_kg_embed/save_embedding_classification.py ./data/hypotheses/embeds
    main()

    
    
    

    