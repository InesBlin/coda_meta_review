# CoDa Narratives

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