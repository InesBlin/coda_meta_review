# -*- coding: utf-8 -*-
"""
Whole pipeline

Data Selection
(1) based on treatment (DataSelector)
(2) data preparation (DataPrep)
(3) based on inclusion criteria (InclusionCriteria)
"""
from typing import Union, Dict
from src.data_selection import DataSelector
from src.data_prep import DataPrep
from src.inclusion_criteria import InclusionCriteria
from src.meta_analysis import MetaAnalysis
from kglab.helpers.data_load import read_csv


def helper_giv(giv):
    """ string to string mapping  """
    return f"{giv.capitalize()}Variable"


def helper_siv(siv):
    """ string to string mapping  """
    siv = siv.split()
    return siv[0] + "".join([x.capitalize() for x in siv[1:]])


class Pipeline:
    """ From data selection to meta-analysis output """
    def __init__(self, giv1, siv1, sivv1, giv2, siv2, sivv2,
                 inclusion_criteria: Union[Dict, None] = None,
                 **cached_moderator):
        self.giv1 = giv1
        self.siv1 = siv1
        self.sivv1 = sivv1
        self.giv2 = giv2
        self.siv2 = siv2
        self.sivv2 = sivv2

        self.data_selector = DataSelector(siv1=siv1, siv2=siv2, sivv1=sivv1, sivv2=sivv2)
        self.data_prep = DataPrep(siv1=siv1, sivv1=sivv1, siv2=siv2, sivv2=siv2)
        self.inclusion_criteria = InclusionCriteria(**inclusion_criteria) \
            if inclusion_criteria else None
        self.meta_analysis = \
            MetaAnalysis(siv1=siv1, sivv1=sivv1,
                         siv2=siv2, sivv2=sivv2, **cached_moderator)

    def get_data_meta_analysis(self, data):
        """ self explanatory """
        data_run = self.data_selector(data=data)
        data_run = self.data_prep(filtered_data=data_run)
        if self.inclusion_criteria is not None:
            data_run = self.inclusion_criteria(data=data_run)
        return data_run

    @staticmethod
    def add_moderator(type_m, options):
        """ Adding moderator from the terminal """
        add_mod = input(f"Do you want to add {type_m} moderators? ")
        if add_mod == "1":
            mods = input(f"Enter all the options you want, separated by ',' from this list: {options} \n")
            mods = mods.split(',')
            return [x for x in mods if x in options]
        return None

    def __call__(self, data, type_rma: str = "uni", es_measure: str = "d",
                 yi: str = "effectSize", method: str = "REML", vi: str = "variance",
                 mods: Union[Dict, None] = None):
        data_run = self.get_data_meta_analysis(data=data)
        data_run.to_csv('data_run.csv')

        output = self.meta_analysis(data=data_run,
            type_rma=type_rma, es_measure=es_measure, yi=yi, method=method, vi=vi,
            mods=mods)
        return output


if __name__ == '__main__':
    DATA = read_csv("./data/observationData.csv")
    VALS = [
        ("punishment", "punishment treatment", "1", "punishment", "punishment treatment", "-1")
    ]
    INCLUSION_CRITERIA = {
        "sample": {"yearOfDataCollection": (1900, 2100)},
        "metadata": {"lang": ["ENG"]},
        "quantitative": {"numberOfObservations": (200, 1000)},
        "study": {"deception": ["FALSE"]},
    }
    CACHED = {
        "study_moderators": "./data/moderators/study_moderators.csv",
        "country_moderators": "./data/moderators/country_moderators.csv",
        "simple_country_moderators": "./data/moderators/simple_country_moderators.csv",
        "complex_country_moderators": "./data/moderators/complex_country_moderators.csv",
        "variable_moderators": "./data/moderators/variable_moderators.csv"
    }
    for GIV1, SIV1, SIVV1, GIV2, SIV2, SIVV2 in VALS:
        PIPELINE = Pipeline(giv1=GIV1, siv1=SIV1, sivv1=SIVV1, giv2=GIV2, siv2=SIV2, sivv2=SIVV2, inclusion_criteria=INCLUSION_CRITERIA, **CACHED)
        CURR_DATA = PIPELINE.get_data_meta_analysis(data=DATA)
        print(CURR_DATA.shape)
        # MODS = ["punishment incentive", "sequential punishment"]
        # MODS = {
        #     "variable": ["punishment incentive"]
        # }
        # MODS = None
        # OUTPUT = PIPELINE(data=DATA, mods=MODS)
        # curr_res = OUTPUT["results_rma"]
        # print(f"{SIV1} : {SIVV1} || {SIV2} : {SIVV2}")
        # print([curr_res[x].reshape((1,))[0] for x in ["b", "k", "pval"]])
        # print("====================")
