# -*- coding: utf-8 -*-
"""
Inclusion criteria
"""
import os
from typing import Union, Dict
import pandas as pd
from src.settings import ROOT_PATH

def check_selection(value, criterion):
    """ Check string selection """
    return [any(x in criterion for x in str(v).split(',')) for v in value]


def helper_range(x, range_val):
    """helper """
    try:
        return float(x) >= range_val[0] and float(x) <= range_val[1]
    except ValueError:
        return False


def check_range(value, range_val):
    """ Check range selection """
    return [helper_range(x, range_val) for x in value]

class InclusionCriteria:
    """ Filtering based on inclusion criteria """
    def __init__(self, sample: Union[Dict, None] = None, study: Union[Dict, None] = None,
                 quantitative: Union[Dict, None] = None, metadata: Union[Dict, None] = None):
        """
        `metadata` can have the following (key, value types):
        - ('authorNames', List)
        - ('lang', List),
        - ('overallN', Tuple[min, max])
        - ('publicationStatus', List)

        - For each element {x} in self.{k}_simple: {k} param can have 
        ({x}, List) as (key, value types)
        - For each element {x} in self.{k}_range: {k} param can have 
        ({x}, Tuple[min, max]) as (key, value types)
        """
        self.sample = sample
        self.study = study
        self.quantitative = quantitative
        self.metadata = metadata

        self.metadata_simple = set(['authorNames', 'publicationStatus', 'lang'])
        self.metadata_range = set(['overallN'])
        self.metadata_range_thresholds = {'overallN': (0, 3000)}
        self.study_simple = set([
            'deception', 'studySymmetric', 'studyKnownEndgame', 'studyDilemmaType',
            'studyContinuousPGG', 'studyExperimentalSetting', 'studyOneShot',
            'studyOneShotRepeated', 'studyMatchingProtocol', 'studyShowUpFee',
            'studyGameIncentive', 'discussion', 'participantDecision',
            'studyRealPartner', 'studyAcquaintance', 'sanction'
        ])
        self.study_range = set([
            'studyNumberOfChoices', 'choiceLow', 'choiceHigh', 'studyKindex',
            'studyMPCR', 'studyGroupSize', 'replenishmentRate',
            'studyPGDThreshold'
        ])
        self.study_range_thresholds = {
            'studyNumberOfChoices': (0, 20),
            'choiceLow': (0, 3),
            'choiceHigh': (0, 15),
            'studyKindex': (0, 1),
            'studyMPCR': (0, 1),
            'studyGroupSize': (0, 400),
            'replenishmentRate': (0, 25),
            'studyPGDThreshold': (0, 25)
        }

        self.sample_simple = set([
            'studyStudentSample', 'studyAcademicDiscipline', 'yearSource', 'country',
            'countrySource', 'recruitmentMethod'
        ])
        self.sample_range = set([
            'yearOfDataCollection', 'meanAge', 'maleProportion', 'ageHigh', 'ageLow',
            'overallN'
        ])
        self.sample_range_thresholds = {
            'yearOfDataCollection': (float('-inf'), float("inf")),
            'meanAge': (0, 100),
            'maleProportion': (0, 1),
            'ageHigh': (0, 100),
            'ageLow': (0, 100),
            'overallN': (0, 3000)
        }

        self.quantitative_simple = set([
            'studyTrialOfCooperation'
        ])
        self.quantitative_range = set([
            'numberOfObservations', 'overallMeanContributions', 'overallStandardDeviation',
            'overallProportionCooperation', 'overallMeanWithdrawal',
            'overallPercentageEndowmentContributed'
        ])
        self.quantitative_range_thresholds = {
            'numberOfObservations': (0, 2500),
            'overallMeanContributions': (0, 15),
            'overallStandardDeviation': (0, 10),
            'overallProportionCooperation': (0, 1),
            'overallMeanWithdrawal': (0, 10),
            'overallPercentageEndowmentContributed': (0, 1)
        }

        self.mappings = {
            "metadata": [self.metadata, self.metadata_range_thresholds,
                         self.metadata_simple, self.metadata_range],
            "study": [self.study, self.study_range_thresholds, self.study_simple, self.study_range],
            "sample": [self.sample, self.sample_range_thresholds,
                       self.sample_simple, self.sample_range],
            "quantitative": [self.quantitative, self.quantitative_range_thresholds,
                             self.quantitative_simple, self.quantitative_range]
        }

    def filter_data(self, data, type_filter):
        """ Apply inclusion criteria to data """
        [attributes, thresholds, simple_set, range_set] = self.mappings[type_filter]
        if attributes:
            for criteria in simple_set.intersection(set(attributes.keys())):
                data = data[check_selection(data[criteria], attributes[criteria])]
                if data.shape[0] == 0:
                    return data
            for criteria in range_set.intersection(set(attributes.keys())):
                (min_n, max_n) = attributes[criteria]
                (min_n_t, max_n_t) = thresholds[criteria]
                if min_n > min_n_t or max_n < max_n_t:
                    data = data[check_range(data[criteria], attributes[criteria])]
                    if data.shape[0] == 0:
                        return data
        return data

    def __call__(self, data: pd.DataFrame):
        """
        In the original server.R code, inclusion criteria:
        - metadata: server.R l690-715 
        - study: server.R l716-883
        - sample: server.R l887-966
        - quantitative: server.R l967-1015
        """
        data["lang"]=data["observationName"].apply(lambda x: x[:3])

        for tf in ["metadata", "study", "sample", "quantitative"]:
            data = self.filter_data(data=data, type_filter=tf)

        return data

def main(ic):
    """ Main """
    data = pd.read_csv(os.path.join(ROOT_PATH, "data/observationData.csv"), index_col=0)
    print(data.shape)
    data = ic(data=data)
    print(data)
    print(data.shape)
    print("===")


if __name__ == '__main__':
    for METADATA in [
        # {"authorNames": ["Bruntsch", "Kopányi-Peuker"]},
        # {"lang": ["JPN", "CHI"]},
        {"overallN": (100, 2300)},
        # {"publicationStatus": ['Published Article', 'Doctoral Dissertation']}
    ]:
        IC = InclusionCriteria(metadata=METADATA)
        main(ic=IC)

    # for STUDY in [
    #     {"deception": ["FALSE"]},
    #     {"studySymmetric": ["FALSE"]},
    #     {"studyKnownEndgame": ["FALSE"]},
    #     {"studyDilemmaType": ["Public Goods Game"]},
    #     {"studyContinuousPGG": ["Continuous"]},
    #     {"studyExperimentalSetting": ['Lab']},
    #     {"studyOneShot": ["Repeated"]},
    #     {"studyOneShotRepeated": ['TRUE']},
    #     {"studyMatchingProtocol": ['Partner']},
    #     {"studyShowUpFee": ['Paid']},
    #     {"studyGameIncentive": "Monetary"},
    #     {"discussion": ["Absent"]},
    #     {"participantDecision": ["Simultaneous"]},
    #     {"studyRealPartner": ["Real"]},
    #     {"studyAcquaintance": ["Strangers"]},
    #     {"sanction": ["TRUE"]},
    #     {"studyNumberOfChoices": (5, 10)},
    #     {"choiceLow": (1, 2)},
    #     {"choiceHigh": (5, 11)},
    #     {"studyKindex": (0.2, 0.7)},
    #     {"studyMPCR": (0.2, 0.7)},
    #     {"studyGroupSize": (100, 223)},
    #     {"replenishmentRate": (5, 20)},
    #     {"studyPGDThreshold": (1, 10)}
    # ]:
    #     IC = InclusionCriteria(study=STUDY)
    #     main(ic=IC)

    # for SAMPLE in [
    #     {"yearOfDataCollection": (1900, 2000)},
    #     {"studyStudentSample": ["FALSE"]},
    #     {"studyAcademicDiscipline": ['Psychology', 'Mixed']},
    #     {"yearSource": ['Conducted', 'Published']},
    #     {"country": ["ITA", "KOR"]},
    #     {"countrySource": ['Specified country', 'All authors']},
    #     {"recruitmentMethod": ['Participant pool', 'Other']},
    #     {"meanAge": (20, 60)},
    #     {"maleProportion": (0.2, 0.4)},
    #     {"ageHigh": (53, 83)},
    #     {"ageLow": (23, 73)},
    #     {"overallN": (1000, 2000)}
    # ]:
    #     IC = InclusionCriteria(sample=SAMPLE)
    #     main(ic=IC)

    # for QUANT in [
    #     {"numberOfObservations": (300, 1000)},
    #     {"overallMeanContributions": (2, 12)},
    #     {"overallStandardDeviation": (5, 7)},
    #     {"overallProportionCooperation": (0.5, 0.78)},
    #     {"overallMeanWithdrawal": (3, 7)},
    #     {"overallPercentageEndowmentContributed": (0.5, 0.78)},
    #     {"studyTrialOfCooperation": ['All trials', 'First trial']}
    # ]:
    #     IC = InclusionCriteria(quantitative=QUANT)
    #     main(ic=IC)
