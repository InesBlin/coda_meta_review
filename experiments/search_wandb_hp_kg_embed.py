# -*- coding: utf-8 -*-
"""
Search hyperparameters for KG embeddings
With wandb + random search
"""
import wandb
from pykeen.pipeline import pipeline
from pykeen.datasets import get_dataset
from src.lp.kg_embedder import KGEmbedder

# Define the sweep configuration
sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'mean_reciprocal_rank',
        'goal': 'maximize'   
    },
    'parameters': {
        'embedding_dim': {'values': [x*16 for x in range(1, 33)]},
        'lr': {'values': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]},
        'num_negs_per_pos': {'values': [1] + [10*x for x in range(1, 11)]},
        'epochs': {'values': [100*x for x in range(1, 6)]},
        'model': {'values': ["rgcn", "distmult", "complex"]},
    }
}

data = "./data/vocab.csv"
subject_col, predicate_col, object_col = "s", "p", "o"
kg_emb = KGEmbedder(
    data_path=data,
    spo_cols=[subject_col, predicate_col, object_col])
wandb.init()

def train():
    wandb.init()
    config = wandb.config
    
    # Run the PyKEEN pipeline with the selected hyperparameters
    result = pipeline(
        model=config.model, random_seed=23,
        training=kg_emb.sh_train, testing=kg_emb.sh_test,
        model_kwargs={'embedding_dim': config.embedding_dim},
        optimizer_kwargs={'lr': config.lr},
        negative_sampler_kwargs={"num_negs_per_pos": config.num_negs_per_pos},
        epochs=config.epochs,
        result_tracker='wandb', result_tracker_kwargs={"project": 'coda-vocab-embeddings'}
        )
    
    metrics = ['hits@1', 'hits@3', 'hits@10', 'mean_rank',
                'mean_reciprocal_rank']
    # Log the results to W&B
    wandb.log({m: result.metric_results.get_metric(m) for m in metrics})

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project='coda-vocab-embeddings')

# Run the sweep
wandb.agent(sweep_id, function=train, count=100)
