# -*- coding: utf-8 -*-
"""
Whole pipeline
"""
from typing import Union, List
from src.data_selection import DataSelector
from src.data_prep import DataPrep
from src.meta_analysis import MetaAnalysis
from kglab.helpers.data_load import read_csv

class Pipeline:
    """ From data selection to meta-analysis output """
    def __init__(self, siv1, sivv1, siv2, sivv2):
        self.siv1 = siv1
        self.sivv1 = sivv1
        self.siv2 = siv2
        self.sivv2 = sivv2

        self.data_selector = DataSelector(siv1=siv1, siv2=siv2, sivv1=sivv1, sivv2=sivv2)
        self.data_prep = DataPrep(siv1=siv1, sivv1=sivv1, siv2=siv2, sivv2=siv2)
        self.meta_analysis = MetaAnalysis(siv1=siv1, sivv1=sivv1, siv2=siv2, sivv2=sivv2)

    def get_data_meta_analysis(self, data):
        """ self explanatory """
        data_run = self.data_selector(data=data)
        data_run = self.data_prep(filtered_data=data_run)
        return data_run

    def __call__(self, data, type_rma: str = "uni", es_measure: str = "d",
                 yi: str = "effectSize", method: str = "REML", vi: str = "variance",
                 mods: Union[List[str], None] = None):
        data_run = self.get_data_meta_analysis(data=data)
        ma_res = self.meta_analysis(data=data_run,
            type_rma=type_rma, es_measure=es_measure, yi=yi, method=method, vi=vi,
            mods=mods)
        return ma_res


if __name__ == '__main__':
    DATA = read_csv("./data/observationData.csv")
    # VALS = [
    #     # (siv1, sivv1, siv2, sivv2)
    #     ("punishment rule", "rank-based", "punishment rule", "contribution-based"),
    #     ("reward incentive", "monetary", "reward incentive", "non-monetary material"),
    #     ("individual difference", "concern for others", "individual difference", "narcissism")
    # ]
    VALS = [
        # (siv1, sivv1, siv2, sivv2)
        # ("gender", "male", "gender", "female"),
        # ("group size level", "low", "group size level", "high"),
        # ("conflict level", "high", "conflict level", "low")
        ("punishment treatment", "1", "punishment treatment", "-1")
    ]
    for siv1, sivv1, siv2, sivv2 in VALS:
        PIPELINE = Pipeline(siv1=siv1, sivv1=sivv1, siv2=siv2, sivv2=sivv2)
        MODS = ["punishment incentive", "sequential punishment"]
        MODS = ["punishment incentive"]
        curr_res = PIPELINE(data=DATA, mods=MODS)
        print(f"{siv1} : {sivv1} || {siv2} : {sivv2}")
        print(curr_res)
        print("====================")