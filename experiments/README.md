# Experiments

This README contains thorough details about the different experiments we ran.

## Folder `anyburl`
All experiments related to AnyBURL (link prediction).
- `final`: contains the final, readable predictions (data, ouptut in .csv and readable formats)
- `preds`: output predictions from AnyBURL
- `rules`: full output, including rules, from AnyBURL
- `analyse_anyburl.ipynb`: statistics and visualisations in the paper
- `evaluate_coda.sh`: bash file for AnyBURL-eval
- `get_anyburl_pred.py`: extract readable outputs from AnyBURL
- `save_data_anyburl.py`: save data in suitable format to run with AnyBURL (explicitly separate train/val/test)
- `train_predict_coda.sh`: bash file for AnyBURL-learn and AnyBURL-apply

## Folder `classification`
All experiments related to classification task (decision trees).
- `final`: contains the final, readable predictions (data, ouptut in .csv and readable formats)
- `hp_search`: results from the hyperparameter search
- `analyse_classification.ipynb`: analyse results of classification
- `run_final_classification.py`: run final classification on test set with best params
- `search_hp_classification.py`: search for best parameters

## Folder `hp_bn_lp`
All experiments related to DL-based link prediction task
- `analyse_data.py`: helper for statistics of data/results (for the paper)
- *_100.{csv,json}: results from random search on (max) 100 random sets of experiments
- `save_data_bn.py`: save data with blank nodes for the link prediction task
- `search_hp_bn_lp.py`: search best parameters

## Folder `hp_kg_embed`
All related to KG embeddings
- `analyse_results.ipynb`: analyse hyperparameter search
- *_378.{csv,json}: results for second grid search params
- *_500.{csv,json}: results for first grid search params
- `save_embedding_classification.py`: save embeddings in `.npy` format for classification
- `search_hp_kg_embed.py`: search best parameters

## Folder `llm_zero_shot_prompting`
All related to LLM method
- `final`: contains the final, readable predictions (data, ouptut in .csv and readable formats)
- `analyse_llm.ipynb`: visualisations/analysis used in the paper
- `run_zero_shot_llm_prompt.py`: run experiment with LLM

## Folder `user_studies`
All related to the user studies
- `cat_moderators.json`: categorisation of moderators
- `hypotheses_form.txt`: readable hypotheses used in the user studies
- `hypotheses.csv`: metadata related to the hypotheses used in the user studies
- `user_studies.ipynb`: choosing hypotheses for the user studies

## Folder `visualisations`
All related to visualisations/statistics
- *.pdf: various plots
- `describe_data.py`: helpers for statistics

## In current directory
- `prep_data.py`: prep data for different methods
- `save_data_hypotheses.py`: extract data from CoDa