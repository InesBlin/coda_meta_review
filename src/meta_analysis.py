# -*- coding: utf-8 -*-
"""
Meta-analyses in Python
"""
from typing import Union, List
import click
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, Formula, NULL, ListVector
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
import pandas as pd
from kglab.helpers.data_load import read_csv
from src.helpers import select_observations

def is_filled(val):
    """ Source: https://github.com/cooperationdatabank/ETL/blob/main/src/convert-indicators.py"""
    emptyVals =[ "", "NA", "NaN" , "N/A" , 'nan' , 'None' , '999' ,'999.0',  'missing' , 'other' , 'others', 'Other', 'Others']
    if str(val) not in emptyVals:
        return True
    return False


def read_data(file_path):
    """ Read .csv and keep columns (all must be filled) """
    df = pd.read_csv(file_path, low_memory=False)
    return df


def convert_pd_to_rdata(input_df: pd.DataFrame):
    """ Convert pandas dataframe to format compatible with R"""
    with (ro.default_converter + pandas2ri.converter).context():
        output = ro.conversion.get_conversion().py2rpy(input_df)
    return output


class MetaAnalysis:
    """ Main class for meta-analysis.
    - Re-use as much as possible of the existing CODA Shiny R application
    https://github.com/cooperationdatabank/rshiny-app
    - R meta-analysis with metafor/rma + rpy2 
    - metafor documentation: https://cran.r-project.org/web/packages/metafor/metafor.pdf"""
    def __init__(self, siv1, sivv1, siv2, sivv2):
        """
        Effect size measure
        - Standardised Mean Difference -> `d`
        - Raw correlation coefficient -> `r`
        """
        self.siv1 = siv1
        self.sivv1 = sivv1
        self.siv2 = siv2
        self.sivv2 = sivv2

        self.metafor = importr("metafor")
        # simple or multilevel model
        self.type_rma = ["uni", "mv"]
        # effect size measure
        self.es_measure = ["d", "r"]

        """character string to specify whether the model should be fitted via 
        maximum likelihood ("ML") or via restricted maximum likelihood ("REML") 
        estimation (the default is "REML")"""
        self.method_mv = ["ML", "REML"]

        """  character string to specify whether an equal- or a random-effects model 
        should be fitted. An equal-effects model is fitted when using method="EE". 
        A random- effects model is fitted by setting method equal to one of the following: 
        "DL", "HE", "HS", "HSk", "SJ", "ML", "REML", "EB", "PM", "GENQ", "PMM", or "GENQM". 
        The default is "REML". See ‘Details’. """
        self.method_uni = ["EE", "DL", "HE", "HS", "HSk", "SJ",
                           "ML", "REML", "EB", "PM", "GENQ", "PMM", "GENQM"]

        self.vars_multilevel_var = ["studyNameGeneral", "paperName"]

        self.vars_res = ["k", "b", "se", "zval", "pval", "ci.lb", "ci.ub", "tau2"]

    def rma_uni(self, yi, vi, data, method, mods, slab):
        """ Simple model """
        if mods:
            mods = Formula("~" + " + ".join([f"`{var}`" for var in mods]))
            return self.metafor.rma_uni(yi=yi, vi=vi, data=data,
                                        method=method, mods=mods, slab=slab)
        return self.metafor.rma_uni(yi=yi, vi=vi, data=data,
                                    method=method, slab=slab)

    def rma_mv(self, yi, V, data, method, random, mods, slab):
        """ Multi-level model """
        if mods:
            mods = Formula("~" + " + ".join([f"`{var}`" for var in mods]))
            return self.metafor.rma_uni(yi=yi, V=V, data=data,
                                        method=method, random=random, mods=mods, slab=slab)
        return self.metafor.rma_uni(yi=yi, V=V, data=data,
                                    method=method, random=random, slab=slab)

    def _check_args(self, type_rma: str, es_measure: str,
                    method, mods: Union[List[str], None], slab: Union[List[str], None],
                    yi: str, data: pd.DataFrame, vi: Union[str, None], V: Union[str, None],
                    multilevel_variables: Union[List[str], None]):
        if type_rma not in self.type_rma:
            raise ValueError(f"`type_rma` must be one of the followings: {self.type_rma}")

        if es_measure not in self.es_measure:
            raise ValueError(f"`es_measure` must be one of the followings: {self.es_measure}")

        if yi not in data.columns:
            raise ValueError(f"yi `{yi}` must be in `data` columns")
        if type_rma == "uni":
            if not vi:
                raise ValueError("If using simple model (type_rma = 'uni'), `vi` must be provided")
            if method not in self.method_uni:
                raise ValueError(f"`method` must be within {self.method_uni}")

        if type_rma == "mv":
            if not (V and multilevel_variables):
                raise ValueError("If using multilevel model (type_rma = 'mv'), " + \
                    "`V` and `multilevel_variables` must be provided")
            if method not in self.method_mv:
                raise ValueError(f"`method` must be within {self.method_mv}")

        if mods:
            for mod in mods:
                if mod not in data.columns:
                    raise ValueError(f"moderator `{mod}` must be in `data` columns")

        if slab and len(slab) != data.shape[0]:
            raise ValueError("`slab` must be a list of labels, with the same size as `data`")


    def get_random(self, multilevel_variables: Union[List[str], None]):
        """ Get random variable necessary for meta-analysis """
        if not multilevel_variables:
            return NULL
        if len(multilevel_variables) == 1:
            return Formula(f"~ 1 | {multilevel_variables[0]}".format)
        if len(multilevel_variables) == 3:
            return ListVector({"paper": Formula("~ 1 | paperName/studyNameGeneral"),
                               "country": Formula("~ 1 | country")})
        if all(var in multilevel_variables for var in self.vars_multilevel_var):
            return Formula("~ 1 | paperName/studyNameGeneral")
        if len(multilevel_variables) == 2:
            return ListVector({f"var{i+1}": Formula(f"~ 1 | {multilevel_variables[i]}") \
                for i in range(2)})
        return NULL

    def filter_data(self, data, es_measure):
        """ Additional filtering """
        data = data[data.effectSizeMeasure == es_measure]

        data = data.groupby('studyNameGeneral').filter(
            lambda x: all(x['substudy'] == 1) or all(x['substudy'] == 0) or \
                (x['substudy'] == 1).any()).reset_index(drop=True)

        so1_1 = select_observations(data, siv=self.siv1, sivv=self.sivv1, treatment_number=1)
        so2_1 = select_observations(data, siv=self.siv1, sivv=self.sivv1, treatment_number=2)
        so1 = np.array(so1_1) == np.array(so2_1)

        so2_2 = select_observations(data, siv=self.siv2, sivv=self.sivv2, treatment_number=1)
        so1_2 = select_observations(data, siv=self.siv2, sivv=self.sivv2, treatment_number=2)
        so2 = np.array(so2_2) == np.array(so1_2)

        filter_ = so1 & so2

        data = data[~np.array(filter_, dtype=bool)]
        return data



    def __call__(self, type_rma: str, es_measure: str,
                 yi: str, data: pd.DataFrame, method: str, mods: Union[List[str], None] = None,
                 vi: Union[str, None] = None, V: Union[str, None] = None,
                 multilevel_variables: Union[List[str], None] = None):
        data = self.filter_data(data=data, es_measure=es_measure)
        slab = list(data.citation.values)

        input_r = convert_pd_to_rdata(input_df=data)
        self._check_args(type_rma=type_rma, es_measure=es_measure, method=method, yi=yi,
                         data=data, mods=mods, slab=slab, vi=vi, V=V,
                         multilevel_variables=multilevel_variables)

        slab = StrVector(slab) if slab else StrVector([""]*data.shape[0])

        if type_rma == "uni":
            res = self.rma_uni(yi=input_r.rx2(yi), vi=input_r.rx2(vi), data=input_r,
                               method=method, mods=mods, slab=slab)
        else:  # type_rma == "mv"
            random = self.get_random(multilevel_variables=multilevel_variables)
            res = self.rma_mv(yi=input_r.rx2(yi), V=input_r.rx2(V), data=input_r, method=method,
                            random=random, mods=mods, slab=slab)
        return {k: res.rx2[k][0] for k in self.vars_res}


@click.command()
# Link to the processed data, .csv format
@click.argument("input_data_path")
def main(input_data_path):
    """ Main script to store obs data """
    meta_analysis = MetaAnalysis(
        siv1="punishment treatment", sivv1="1",
        siv2="punishment treatment", sivv2="-1")
    data = read_csv(input_data_path)
    results_rma = meta_analysis(type_rma="uni", es_measure="d", yi="effectSize", data=data,
                                method="REML", vi="variance")
    print(results_rma)


if __name__ == '__main__':
    # python src/meta_analysis.py data/observationData.csv
    main()
