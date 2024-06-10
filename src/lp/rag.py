# -*- coding: utf-8 -*-
"""
RAG-based baseline
"""

import os
import pandas as pd
from tqdm import tqdm
from loguru import logger
from llama_index.core.schema import Document, Node
from llama_index.core import KnowledgeGraphIndex
from src.settings import API_KEY_GPT

DC = 'http://purl.org/dc/terms/'
CDP = 'https://data.cooperationdatabank.org/vocab/prop/'
RDF = 'http://www.w3.org/2000/01/rdf-schema#'
SKOS = "http://www.w3.org/2004/02/skos/core#"
os.environ["OPENAI_API_KEY"] = API_KEY_GPT

def get_label(x):
    if "#" in x:
        return x.split("#")[-1]
    return x.split("/")[-1]

def get_node_text(node, df, pred_to_label):
    """ Retrieving node's text by taking the literals """
    curr_df = df[(df.subject==node) & (~df.object.str.startswith("http"))]

    if f"{RDF}label" in curr_df.predicate.values:
        subject_label = curr_df[curr_df.predicate==f"{RDF}label"].object.values[0]
    else:
        subject_label = node.split('/')[-1]
    return "\n".join([f"{subject_label} {pred_to_label[row.predicate]} {row.object}" \
        for _, row in curr_df.iterrows()])

class RAGBasedLP:
    """ Generating hypotheses with RAG """
    def __init__(self, data: str):
        df = pd.read_csv(data).dropna()
        self.metrics = {"triples_init": df.shape[0]}

        df = df[(df.subject.str.startswith("http")) & (df.predicate.str.startswith("http"))]
        self.metrics.update({"triples_subj_pred_iri": df.shape[0]})
        self.df = df
        self.pred_labels = self.get_pred_labels()
    
    def get_pred_labels(self):
        """ Get labels of predicates """
        all_predicates = self.df.predicate.unique()
        self.metrics.update({"nb_pred": all_predicates.shape[0]})

        df_preds_with_labels = self.df[(self.df.subject.isin(all_predicates)) & (self.df.predicate==f"{RDF}label")]
        self.metrics.update({"nb_pred_labels": df_preds_with_labels.shape[0]})

        pred_labels = df_preds_with_labels.set_index('subject')['object'].to_dict()

        remaining_preds = list(set(all_predicates).difference(set(df_preds_with_labels.subject.unique())))

        pred_labels.update({x: get_label(x) for x in remaining_preds})
        return pred_labels
    
    def get_nodes(self):
        """ Get nodes and their descriptions """
        all_nodes = self.df.subject.unique()
        self.metrics.update({"nb_nodes": all_nodes.shape[0]})

        nodes = []
        llama_nodes = []
        logger.info("Adding nodes")
        for node in tqdm(all_nodes):
            text = get_node_text(node, self.df, self.pred_labels)
            llama_nodes.append(Node(text=text, id_=node))
            nodes.append((node, text))
        df = pd.DataFrame(nodes, columns=["id_", "text"])
        return all_nodes, llama_nodes, df
    
    def __call__(self, save_dir):
        """ Save index """
        all_nodes, llama_nodes, df = self.get_nodes()
        df.to_csv(os.path.join(save_dir, "node_text.csv"))
        logger.info("Building index")
        index = KnowledgeGraphIndex(nodes=llama_nodes)
        index.storage_context.persist(save_dir)
        for _, row in self.df[(self.df.subject.isin(all_nodes)) & \
            (self.df.object.str.startswith('http'))].iterrows():
            index.upsert_triplet((row.subject, row.predicate, row.object))
        index.storage_context.persist(save_dir)

    
if __name__ == '__main__':
    DATA = "./data/coda_kg.csv"
    RBLP = RAGBasedLP(data=DATA)
    RBLP(save_dir="experiments/rag")