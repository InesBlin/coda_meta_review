# -*- coding: utf-8 -*-
"""
Whole pipeline
"""
from typing import Union, List
from src.data_selection import DataSelector
from src.data_prep import DataPrep
from src.meta_analysis import MetaAnalysis
from kglab.helpers.data_load import read_csv

def helper_giv(giv):
    return f"{giv.capitalize()}Variable"

def helper_siv(siv):
    siv = siv.split()
    return siv[0] + "".join([x.capitalize() for x in siv[1:]])

class Pipeline:
    """ From data selection to meta-analysis output """
    def __init__(self, giv1, siv1, sivv1, giv2, siv2, sivv2):
        self.giv1 = giv1
        self.siv1 = siv1
        self.sivv1 = sivv1
        self.giv2 = giv2
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

    @staticmethod
    def add_moderator(type_m, options):
        add_mod = input(f"Do you want to add {type_m} moderators? ")
        if add_mod == "1":
            mods = input(f"Enter all the options you want, separated by ',' from this list: {options} \n")
            mods = mods.split(',')
            return [x for x in mods if x in options]
        return None

    def __call__(self, data, type_rma: str = "uni", es_measure: str = "d",
                 yi: str = "effectSize", method: str = "REML", vi: str = "variance",
                 mods: Union[List[str], None] = None):
        data_run = self.get_data_meta_analysis(data=data)

        # if not mods:
        #     add_mods = input("You have not entered any moderator, do you want to add some? ")
        #     if add_mods == '1':
        #         mods = {}
        #         variable_mods = list(self.meta_analysis.moderator.get_variable_moderators(
        #             data=data_run, info={"giv1": helper_giv(self.giv1), "siv1": helper_siv(self.siv1),
        #                                  "giv2": helper_giv(self.giv2), "siv2": helper_siv(self.siv2)}))
        #         for type_m, options in [
        #             ("variable", variable_mods),
        #             ("study", self.meta_analysis.moderator.study_moderators),
        #             ("country", self.meta_analysis.moderator.country_moderators)]:
        #             res = self.add_moderator(type_m=type_m, options=options)
        #             if res:
        #                 mods[type_m] = res

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
        ("punishment", "punishment treatment", "1", "punishment", "punishment treatment", "-1")
    ]
    for giv1, siv1, sivv1, giv2, siv2, sivv2 in VALS:
        PIPELINE = Pipeline(giv1=giv1, siv1=siv1, sivv1=sivv1, giv2=giv2, siv2=siv2, sivv2=sivv2)
        MODS = ["punishment incentive", "sequential punishment"]
        MODS = {
            "variable": ["punishment incentive"]
        }
        curr_res = PIPELINE(data=DATA, mods=MODS)
        print(f"{siv1} : {sivv1} || {siv2} : {sivv2}")
        print(curr_res)
        print("====================")
