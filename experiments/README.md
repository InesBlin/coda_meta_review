# Experiments

This README contains thorough details about the different experiments we ran. We hereafter provide information about the data and the scripts to run for reproducibility of our experiments.

## Starting scripts
You first need to retrieve data from the original KG, and to format your data for the right method. Each script will save and format data for all three types of hypothesis.

### Available data
For ease of access, we make the processed data directly available in the `data.zip` zipped file. To unzip (from directory):
```bash
unzip data.zip
```

The starting scripts creates the following folders with content: `data/hypotheses/entry`, `data/hypotheses/llm`, `data/hypotheses/classification` and `data/hypotheses/lp`.

### Retrieve the data yourself

It is possible to automatically retrieve the data by yourself. To retrieve data from the KG, you first need to set up a local repository of CoDa. We used the GraphDB software. The graphs can be uploaded on [this website](https://odissei.triply.cc/coda/databank/graphs).

First retrieve the data from the local endpoint:
```bash
python experiments/save_data_hypotheses.py <your-local-endpoint> experiments/cat_moderators.json data/hypotheses/entry
```
This will run API calls to `<your-local-endpoint>`, categorise moderators and save them in `experiments/cat_moderators.json` and save the data in the `data/hypotheses/entry` folder. The arguments can be changed.

Then format the data for each method:
```bash
python experiments/prep_data.py data/hypotheses/entry/ data/hypotheses/llm llm
python experiments/prep_data.py data/hypotheses/entry/ data/hypotheses/classification classification
python experiments/prep_data.py data/hypotheses/entry/ data/hypotheses/lp lp
```
This will format the data from the folder data/hypotheses/entry/ and save it into the output folder, here `data/hypotheses/llm`, `data/hypotheses/classification` and `data/hypotheses/lp`. 

## Hypothesis Generation
We hereafter detail the subsequent scripts/data for each of the methods. Please check each script independently for examples of terminal command.

### Classification task

To reproduce the experiments:
- `hp_kg_embed/search_hp_kg_embed.py` to search the best parameters for embeddings
- `hp_kg_embed/save_embedding_classification.py` to save the best embeddings
- `classification/search_hp_classification.py` to search the best parameters for the classification
- `hp_kg_embed/run_final_classification.py` to save results on the best classification model

#### Folder `hp_kg_embed`
All related to KG embeddings
- `analyse_results.ipynb`: analyse hyperparameter search
- *_378.{csv,json}: results for second grid search params
- *_500.{csv,json}: results for first grid search params
- `save_embedding_classification.py`: save embeddings in `.npy` format for classification
- `search_hp_kg_embed.py`: search best parameters

#### Folder `classification`
All experiments related to classification task (decision trees).
- `final`: contains the final, readable predictions (data, ouptut in .csv and readable formats)
- `hp_search`: results from the hyperparameter search
- `analyse_classification.ipynb`: analyse results of classification
- `run_final_classification.py`: run final classification on test set with best params
- `search_hp_classification.py`: search for best parameters



### Link prediction task
This task contains two different types of methods: symbolic with AnyBURL (cf. folder `anyburl`), and sub-symbolic with regular DL models (cf. folder `hp_bn_lp`).

#### Folder `anyburl`
All experiments related to AnyBURL (link prediction). You can download AnyBURL (and see further instructions) on [this website](https://web.informatik.uni-mannheim.de/AnyBURL/).
- `final`: contains the final, readable predictions (data, ouptut in .csv and readable formats)
- `preds`: output predictions from AnyBURL
- `rules`: full output, including rules, from AnyBURL
- `analyse_anyburl.ipynb`: statistics and visualisations in the paper
- `evaluate_coda.sh`: bash file for AnyBURL-eval
- `get_anyburl_pred.py`: extract readable outputs from AnyBURL
- `save_data_anyburl.py`: save data in suitable format to run with AnyBURL (explicitly separate train/val/test)
- `train_predict_coda.sh`: bash file for AnyBURL-learn and AnyBURL-apply


#### Folder `hp_bn_lp`
All experiments related to DL-based link prediction task
- `analyse_data.py`: helper for statistics of data/results (for the paper)
- *_100.{csv,json}: results from random search on (max) 100 random sets of experiments
- `save_data_bn.py`: save data with blank nodes for the link prediction task
- `search_hp_bn_lp.py`: search best parameters

### LLM task

#### Folder `llm_zero_shot_prompting`
All related to LLM method
- `final`: contains the final, readable predictions (data, ouptut in .csv and readable formats)
- `analyse_llm.ipynb`: visualisations/analysis used in the paper
- `run_zero_shot_llm_prompt.py`: run experiment with LLM

- 

## User Studies

### Folder `user_studies`
All related to the user studies
- `cat_moderators.json`: categorisation of moderators
- `hypotheses_form.txt`: readable hypotheses used in the user studies
- `hypotheses.csv`: metadata related to the hypotheses used in the user studies
- `user_studies.ipynb`: choosing hypotheses for the user studies

## Misc
### Folder `visualisations`
All related to visualisations/statistics
- *.pdf: various plots
- `describe_data.py`: helpers for statistics

