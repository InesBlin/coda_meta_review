# -*- coding: utf-8 -*-
"""
Generating hypotheses based on literature
"""
import random
from typing import Union
from src.hg.hg import HypothesisGenerator


class LiteratureHypothesisGenerator(HypothesisGenerator):
    """ Proposing hypotheses based on literature """
    def __init__(self):
        """ Storing hypotheses """
        self.regular_h = [
            # Paper: Gender differences in cooperation across 20 societies: a meta-analysis
            {"giv1": "gender", "siv1": "gender", "sivv1": "female",
             "giv2": "gender", "siv2": "gender", "sivv2": "male",
             'reg_qualifier': 'higher'}
        ]
        self.numerical_h = [
            # Paper: Gender differences in cooperation across 20 societies: a meta-analysis
            {"giv1": "gender", "siv1": "gender", "sivv1": "female",
             "giv2": "gender", "siv2": "gender", "sivv2": "male",
             'num_qualifier': 'higher', 'mod_qualifier': 'higher',
             'mod': 'studyKindex', 'type_mod': 'study'}
        ]
        self.categorical_h = [
            # Paper: Gender differences in cooperation across 20 societies: a meta-analysis
            {"giv1": "gender", "siv1": "gender", "sivv1": "female",
             "giv2": "gender", "siv2": "gender", "sivv2": "male",
             'cat_qualifier': 'higher', 'mod_qualifier': 'higher',
             'mod': 'studyOneShot', 'type_mod': 'study',
             'mod1': 'Repeated', 'mod2': 'One-shot'}
        ]
        self.h = self.regular_h + self.numerical_h + self.categorical_h

    def __call__(self, giv: Union[str, None] = None, top_k: int = 1):
        if giv:
            hypotheses = [h for h in self.h if h.get('giv1') == giv]
        else:
            hypotheses = self.h

        if not hypotheses:
            return None

        top_k = min(top_k, len(hypotheses))
        indexes = random.sample(list(range(len(hypotheses))), top_k)
        return [hypotheses[i] for i in indexes]


if __name__ == '__main__':
    LITERATURE_HG = LiteratureHypothesisGenerator()
    print("Random: ", LITERATURE_HG())
    print("Random + gender: ", LITERATURE_HG(giv="gender"))
    print("Random + personality: ", LITERATURE_HG(giv="personality"))
