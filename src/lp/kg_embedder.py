# -*- coding: utf-8 -*-
"""
KG Embedder
"""
import click
from typing import List, Union, Dict
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from pykeen.pipeline import pipeline
from pykeen.hpo import hpo_pipeline
from pykeen.triples import TriplesFactory
from pykeen.nn.init import PretrainedInitializer

def get_description_node(x, helpers, cols):
    """ Get a description from one of the following predicates:
    (1) description (2) description iv (3) title (4) label """
    for type_des in ["description", "description_iv", "title", "label"]:
        df_ = helpers[type_des]
        if x in df_[cols[0]].unique():
            des = df_[df_[cols[0]]==x][cols[2]].values[0]
            if isinstance(des, str):
                return des
    # No description found
    return x

def embed_descriptions(descriptions,
                       model_name="sentence-transformers/all-MiniLM-L6-v2",
                       batch_size=64):
    """ Embed transformers using pre-trained huggingface model """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    tokenized_texts = tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt")
    model.eval()

    # Generate embeddings in batches
    with torch.no_grad(), tqdm(total=len(descriptions)) as progress_bar:
        embeddings = []
        for i in range(0, len(descriptions), batch_size):
            batch_tokenized_texts = {key: value[i:i+batch_size] \
                for key, value in tokenized_texts.items()}
            outputs = model(**batch_tokenized_texts)
            # Mean pooling over token embeddings
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings)
            progress_bar.update(len(batch_embeddings))

    return torch.cat(embeddings, dim=0)


class KGEmbedder:
    """ 
    - Embedding KG nodes
    - Possibility to initialise from pre-trained embeddings
    """
    def __init__(self, data_path: str, spo_cols: List[str], create_inverse_triples: bool = False):
        """
        - data_path: .csv path with data
        - spo_cols: names of subject, predicate, object cols
        """
        self.spo_cols = spo_cols
        self.dc = 'http://purl.org/dc/terms/'
        self.cdp = 'https://data.cooperationdatabank.org/vocab/prop/'
        self.filter_out_pred = [
            'http://www.w3.org/2000/01/rdf-schema#label',
            self.dc + 'description',
            self.dc + 'title',
            self.cdp + 'descriptionIV',
            'http://www.w3.org/2002/07/owl#sameAs',
            self.cdp + "otherVariables",
            # Below: DOI/volume/page/issn-related
            "http://purl.org/ontology/bibo/doi",
            "http://prismstandard.org/namespaces/basic/2.1/doi",
            "http://purl.org/dc/terms/identifier",
            "http://purl.org/ontology/bibo/volume",
            "http://prismstandard.org/namespaces/basic/2.1/volume",
            "http://prismstandard.org/namespaces/basic/2.1/startingPage",
            "http://purl.org/ontology/bibo/pageStart",
            "http://prismstandard.org/namespaces/basic/2.1/endingPage",
            "http://purl.org/ontology/bibo/pageEnd",
            "http://purl.org/ontology/bibo/issn",
            "http://prismstandard.org/namespaces/basic/2.1/issn"
        ]
        self.filter_out_indicators = [
            self.cdp + 'value',
            self.cdp + 'year',
        ]
        self.filter_out_if_lit = [
            self.cdp + "PersonalityVariable"
        ]
        df = pd.read_csv(data_path)

        self.helpers = {key: df[df[self.spo_cols[1]] == val] \
            for (key, val) in [
                ("description", self.dc + 'description'),
                ("description_iv", self.cdp + 'descriptionIV'),
                ("title", self.dc + 'title'),
                ("label", 'http://www.w3.org/2000/01/rdf-schema#label')
            ]}

        df = df[~df[spo_cols[1]].isin(self.filter_out_pred + self.filter_out_indicators)]
        df = df[~((df[spo_cols[1]].isin(self.filter_out_if_lit)) & \
            (df[spo_cols[2]].str.startswith("http")))]
        df = df.dropna(subset=spo_cols)

        # Only keeping IRIs
        df = df[df[spo_cols[2]].str.startswith("http")]
        self.df = df

        # TO-REMOVE: sampling
        # self.df = self.df.sample(n=50000, random_state=23)

        self.sh = TriplesFactory.from_labeled_triples(
            self.df[spo_cols].values,
            create_inverse_triples=create_inverse_triples)
        self.sh_train, self.sh_test = self.sh.split([0.9, 0.1], random_state=23)
        self.model = None

    def get_description(self, entity_id_to_label):
        """ Retrieve, when possible, the description """
        df = pd.DataFrame.from_dict(entity_id_to_label, orient='index')
        df.columns = ["label"]

        tqdm.pandas()
        df["description"] = df["label"].progress_apply(
            lambda x: get_description_node(x, self.helpers, self.spo_cols))
        return df

    def get_embeddings(self, entity_id_to_label: Union[Dict, None] = None,
                       df_description: Union[pd.DataFrame, None] = None,
                       save_path: Union[str, None] = None):
        """ Embeddings of nodes """
        if not isinstance(df_description, pd.DataFrame):
            if not entity_id_to_label:
                raise ValueError("If you don't give the descriptions you need to specify `entity_id_to_label`.")
            df_description = self.get_description(entity_id_to_label)
        descriptions = df_description.description.tolist()
        embeddings = embed_descriptions(descriptions=descriptions)
        if save_path:
            torch.save(embeddings, save_path)
        
    def search_hpo(self, model_kwargs_ranges: dict, model: str = "rgcn"):
        """ Hyperparameter search """
        output = hpo_pipeline(
            model=model,
            training=self.sh_train, testing=self.sh_test,
            model_kwargs_ranges=model_kwargs_ranges
        )
        return output

    def init_pipeline(self, model: str = "transe", random_seed: int = 23,
                      epochs: int = 250, embedding_dim: int = 256,
                      lr: float = 0.01, num_negs_per_pos: int = 50,
                      embeddings: Union[torch.Tensor, None] = None):
        """ Init pykeen pipeline """
        if isinstance(embeddings, torch.Tensor):
            model_kwargs = {
                "embedding_dim": embeddings.shape[-1],
                "entity_initializer": PretrainedInitializer(tensor=embeddings)
            }
        else:
            model_kwargs = {"embedding_dim": embedding_dim}
        output = pipeline(
            model=model, random_seed=random_seed,
            training=self.sh_train, testing=self.sh_test,
            model_kwargs=model_kwargs,
            optimizer_kwargs={"lr": lr},
            negative_sampler_kwargs={"num_negs_per_pos": num_negs_per_pos},
            epochs=epochs
            # training_loop='sLCWA',
            # training_kwargs=dict(batch_size=64, learning_rate=0.1, epochs=epochs)
            )
        self.model = output.model
        # pipeline.save_to_directory('folder_pipeline')
        # model in .pkl file, then model.entity_representations
        return output
    
@click.command()
@click.argument("data")
@click.argument("subject_col")
@click.argument("predicate_col")
@click.argument("object_col")
@click.argument("model")
@click.argument("epochs")
@click.argument("embedding_dim")
@click.argument("lr")
@click.argument("num_negs_per_pos")
@click.argument("save")
def main(data, subject_col, predicate_col, object_col, model,
         epochs, embedding_dim, lr, num_negs_per_pos, save):
    kg_emb = KGEmbedder(data_path=data, spo_cols=[subject_col, predicate_col, object_col], create_inverse_triples=False)
    pipeline = kg_emb.init_pipeline(
        model=model, random_seed=23, epochs=int(epochs), embedding_dim=int(embedding_dim),
        lr=float(lr), num_negs_per_pos=int(num_negs_per_pos)
    )
    for key in ['hits@1', 'hits@3', 'hits@10', 'mean_reciprocal_rank']:
        print(f"{key}: {pipeline.metric_results.get_metric(key)}")
    pipeline.save_to_directory(save)



if __name__ == '__main__':
    # python src/lp/kg_embedder.py ./data/vocab.csv s p o distmult 500 336 0.001 1 models/coda_ontology
    main()

 