# CoDa Meta Reviews

This is the code to the interface that we submitted together with the paper "Living Meta-Review Generation for Social Scientists: An Interface and A Case Study on Human Cooperation", accepted as a demo paper to [EKAW 2024](https://event.cwi.nl/ekaw2024/index.html). The code contains the followings:
- The core code in the `src` folder, for general usage;
- An interface and templates for the meta-review automated generation, cf. folders `app` and `meta_review`.


## Installing a virtualenv + Set-Up

We used Python 3.10.8. 

First create a file `src/settings.py` and add the root directory:
```python
ROOT_PATH = "/your/root/path/"
```

Installing rpy2 can cause problems with conda, hence we recommend to first install it before installing all the other dependencies.

```bash
conda install rpy2
pip install -r requirements.txt
python setup.py install
git clone git@github.com:SonyCSLParis/kglab.git
cd kglab && python setup.py install
```

## Rune the Interface
You first need to unzip the data that is cached for optimisation:
```bash
unzip data.zip
```

Then you can run the app from the root directory:
```bash
streamlit run app/Home.py
```

## Adapting the R shiny app
We have adapted code from the original R shiny app, and transferred it to Python: https://github.com/cooperationdatabank/rshiny-app. This includes the followings:
* `src/get_obs_data.py`: retrieve observation data
* `src/data_selection.py`: takes observation data as input, outputs subset of this data
    * For now: based on comparison between two treatments (specific independent variables and their value)
* `src/data_prep.py`: takes pre-selected data as input, and processes it for the meta-analysis
* `src/meta_analysis.py`: takes processed data as input, and does the meta-analysis

## Data
To make the app more easibly runnable, we make the data directly available.

However, we also made sure that the data collection part was reproducible, for better research practices and to enable regular data updates (for instance, if CoDa is updated)

To retrieve data from the KG, you first need to set up a local repository of CoDa. We used the GraphDB software. The graphs can be uploaded on [this website](https://odissei.triply.cc/coda/databank/graphs). You could also use the public CoDA API, but there might be query limitations.

To make the application more efficient, we first cached some data:

- On the moderators:
    - Caching: cf. `src/helpers/cache_data.py` with examples to run from terminal command
    - Categorising: cf. `src/helpers/categorise_moderators.py` with examples to run from terminal command
- On observation data used for the interface (also directly available):
    - `src/helpers/get_obs_data.py`
- On the variables:
    - `src/helpers/get_generic_specific.py`

## Code structure
- `app`: streamlit application
- `experiments`: various early experiments we ran (not useful for the interface)
- `meta_review`: HTML template for the meta-review
- `src`: source code
    - `configs`: example configs for the meta-review
    - `helpers`: described above, and if not: generic helpers
    - (`hg`: hypothesis generation to integrate in the interface)
    - (`visualisations`: visualisations from early experiments)
    - `data_prep.py`: format data for meta-analysis
    - `data_selection.py`: select subset of observation data
    - `inclusion_criteria.py`: inclusion criteria for studies
    - `knowledge.py`: generating human-readable hypotheses
    - `meta_analysis.py`: meta analysis with `rpy2`
    - `meta_review.py`: meta review generation
    - `moderator.py`: all related to moderators
    - `pipeline.py`: end-to-end process

## Acknowledgements

If you use this work please cite the following paper:
```bib
{
    to add when proceedings are published
}
```

This work was partly funded by the European
[MUHAI](https://muhai.org) project, grant no. 951846, and by the XS NWO Project, grant no. [406.XS.04.118](https://www.nwo.nl/en/projects/406xs04118). We also thank our reviewers for constructive comments.
