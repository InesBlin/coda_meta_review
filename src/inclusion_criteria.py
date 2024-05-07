# -*- coding: utf-8 -*-
"""
Inclusion criteria
"""
import re
from typing import Union, Dict, List, Tuple
import pandas as pd

def check_selection(value, criterion):
    """ Check string selection """
    return [any(x in criterion for x in str(v).split(',')) for v in value]

# def check_range(value, range_val):
#     """ Check range selection """
#     return [any(float(x) >= range_val[0] and float(x) <= range_val[1] for x in v.split(',')) for v in value]


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
        """
        self.sample = sample
        self.study = study
        self.quantitative = quantitative
        self.metadata = metadata

        self.metadata_simple = set(['authorNames', 'publicationStatus'])
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

        ])
        self.sample_range = set([
            'yearOfDataCollection'
        ])
        self.sample_range_thresholds = {
            'yearOfDataCollection': (float('-inf'), float("inf"))
        }

    def __call__(self, data: pd.DataFrame):
        if self.metadata:  # server.R l690-715
            if 'lang' in self.metadata:
                data = data[data['observationName'].str.startswith(tuple(self.metadata['lang']))]
            if 'overallN' in self.metadata:
                (min_n, max_n) = self.metadata['overallN']
                if min_n > 0 or max_n < 3000:
                    data = data[check_range(data['overallN'], self.metadata['overallN'])]
            for criteria in self.metadata_simple.intersection(set(self.metadata.keys())):
                data = data[check_selection(data[criteria], self.metadata[criteria])]

        if self.study:  # server.R l716-883
            for criteria in self.study_simple.intersection(set(self.study.keys())):
                data = data[check_selection(data[criteria], self.study[criteria])]
            for criteria in self.study_range.intersection(set(self.study.keys())):
                (min_n, max_n) = self.study[criteria]
                (min_n_t, max_n_t) = self.study_range_thresholds[criteria]
                if min_n > min_n_t or max_n < max_n_t:
                    data = data[check_range(data[criteria], self.study[criteria])]
        
        if self.sample:
            for criteria in self.sample_simple.intersection(set(self.sample.keys())):
                data = data[check_selection(data[criteria], self.sample[criteria])]
            for criteria in self.sample_range.intersection(set(self.sample.keys())):
                (min_n, max_n) = self.sample[criteria]
                (min_n_t, max_n_t) = self.sample_range_thresholds[criteria]
                if min_n > min_n_t or max_n < max_n_t:
                    data = data[check_range(data[criteria], self.sample[criteria])]

        return data
    
def main(ic):
    """ Main """
    data = pd.read_csv("./data/observationData.csv", index_col=0)
    print(data.shape)
    data = ic(data=data)
    print(data)
    print(data.shape)
    print("===")


if __name__ == '__main__':
    for METADATA in [
        # {"authorNames": ["Bruntsch", "Kopányi-Peuker"]},
        # {"lang": ["JPN", "CHI"]},
        # {"overallN": (100, 2300)},
        # {"publicationStatus": ['Published Article', 'Doctoral Dissertation']}
    ]:
        IC = InclusionCriteria(metadata=METADATA)
        main(ic=IC)

    for STUDY in [
        #{"deception": ["FALSE"]},
        # {"studySymmetric": ["FALSE"]},
        # {"studyKnownEndgame": ["FALSE"]},
        # {"studyDilemmaType": ["Public Goods Game"]},
        # {"studyContinuousPGG": ["Continuous"]},
        # {"studyExperimentalSetting": ['Lab']},
        # {"studyOneShot": ["Repeated"]},
        # {"studyOneShotRepeated": ['TRUE']},
        # {"studyMatchingProtocol": ['Partner']},
        # {"studyShowUpFee": ['Paid']},
        # {"studyGameIncentive": "Monetary"},
        # {"discussion": ["Absent"]},
        # {"participantDecision": ["Simultaneous"]},
        # {"studyRealPartner": ["Real"]},
        # {"studyAcquaintance": ["Strangers"]},
        # {"sanction": ["TRUE"]},
        # {"studyNumberOfChoices": (5, 10)},
        # {"choiceLow": (1, 2)},
        # {"choiceHigh": (5, 11)},
        # {"studyKindex": (0.2, 0.7)},
        # {"studyMPCR": (0.2, 0.7)},
        # {"studyGroupSize": (100, 223)},
        # {"replenishmentRate": (5, 20)},
        # {"studyPGDThreshold": (1, 10)}
    ]:
        IC = InclusionCriteria(study=STUDY)
        main(ic=IC)

    for SAMPLE in [
        {"yearOfDataCollection": (1900, 2000)}
    ]:
        IC = InclusionCriteria(sample=SAMPLE)
        main(ic=IC)

"""
up to l892

"""