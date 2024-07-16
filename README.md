# Automated Hypothesis Generation on Human Cooperation

This is the code we submit together with the paper [TO-ADD], submitted to [TO-ADD]. The code contains the followings:
- The core code in the `src` folder, for general usage
- All elements related to our experiments, including:
    * The data and models, cf. folders `data/zip` and `models`
    * Thorough details on the experiements in the folder `experiments`. For more clarity we provide a README.md for the experiments in that folder.
- An interface and templates for the meta-review automated generation, cf. folders `app` and `meta_review`.


## Installing a virtualenv + Set-Up

We used Python 3.10.8. If you plan to use the OpenAI API, you need to add your API key. To do, create a file `src/settings.py` and add your key:
```python
API_KEY_GPT = "your-key
```

Installing rpy2 can cause problems with conda, hence we recommend to first install it before installing all the other dependencies.

```bash
conda install rpy2
pip install -r requirements.txt
python setup.py install
cd kglab && python setup.py install
```

## Adapting the R shiny app

Aim = transfer the R code to Python, to make it easily runnable

* `src/get_obs_data.py`: retrieve observation data
* `src/data_selection.py`: takes observation data as input, outputs subset of this data
    * For now: based on comparison between two treatments (specific independent variables and their value)
* `src/data_prep.py`: takes pre-selected data as input, and processes it for the meta-analysis
* `src/meta_analysis.py`: takes processed data as input, and does the meta-analysis

## Interface
To retrieve data from the KG, you first need to set up a local repository of CoDa. We used the GraphDB software. The graphs can be uploaded on [this website](https://odissei.triply.cc/coda/databank/graphs).

To make the application more efficient, we first cached some data:

- On the moderators:
    - Caching: cf. `src/helpers/cache_data.py` with examples to run from terminal command
    - Categorising: cf. `src/helpers/categorise_moderators.py` with examples to run from terminal command
- On observation data used for the interface (also directly available):
    - `src/helpers/get_obs_data.py`


Depreciated (not used anymore):
- `src/helpers/get_generic_specific.py`