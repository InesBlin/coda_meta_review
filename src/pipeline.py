# -*- coding: utf-8 -*-
"""
Whole pipeline
"""
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

    def __call__(self, data, type_rma: str = "uni", es_measure: str = "d",
                 yi: str = "effectSize", method: str = "REML", vi: str = "variance"):
        data_run = self.data_selector(data=data)
        data_run = self.data_prep(filtered_data=data_run)
        print(data_run.shape)
        ma_res = self.meta_analysis(data=data_run,
            type_rma=type_rma, es_measure=es_measure, yi=yi, method=method, vi=vi)
        return ma_res
        return None


if __name__ == '__main__':
    DATA = read_csv("./data/observationData.csv")
    VALS = [
        # (siv1, sivv1, siv2, sivv2)
        ("punishment rule", "rank-based", "punishment rule", "contribution-based"),
        ("reward incentive", "monetary", "reward incentive", "non-monetary material"),
        ("individual difference", "concern for others", "individual difference", "narcissism")
    ]
    for siv1, sivv1, siv2, sivv2 in VALS:
        PIPELINE = Pipeline(siv1=siv1, sivv1=sivv1, siv2=siv2, sivv2=sivv2)
        curr_res = PIPELINE(data=DATA)
        print(f"{siv1} : {sivv1} || {siv2} : {sivv2}")
        print(curr_res)
        print("====================")