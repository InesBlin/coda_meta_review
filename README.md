# CoDa Narratives

## Experiments

`experiments` folder

1. Extract data to use
* save_data_hypotheses.py

2. Prep data
* prep_data.py

3. Models

* Classification
    * search_hp_kg_embed.py: search hyperparams for kg embeddings
    * save_embedding_classification.py: saving embeddding to train models
    * search_hp_classification.py: search hyperparams for classification
    * run_final_classification.py: run final models for classification task + readable output
* LLM
    * run_zero_shot_llm_prompt.py: run prompting for ontology-based prompts
* LP
    * search_hp_bn_lp.py: search hyperparams for link prediction
    * save_data_bn.py: build KG for LP task


Folders:
* classification: results of classification models
* hp_kg_embed: results of hyperparameter search for kg embeddings (ontology)
* llm_zero_shot_prompting: results of llm 

## Adapting the R shiny app

Aim = transfer the R code to Python, to make it easily runnable

* `src/get_obs_data.py`: retrieve observation data
* `src/data_selection.py`: takes observation data as input, outputs subset of this data
    * For now: based on comparison between two treatments (specific independent variables and their value)
* `src/data_prep.py`: takes pre-selected data as input, and processes it for the meta-analysis
* `src/meta_analysis.py`: takes processed data as input, and does the meta-analysis
* ``


## Installing a virtualenv

We used Python 3.10.8

Installing rpy2 can cause problems with conda, hence we recommend to first install it before installing all the other dependencies.

```bash
conda install rpy2
pip install -r requirements.txt
python setup.py install
```