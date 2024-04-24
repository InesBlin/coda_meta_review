# -*- coding: utf-8 -*-
"""
Generating hypotheses based on LLM prompting
"""
import random
from typing import Union
import pandas as pd
import numpy as np
from src.hg.hg import HypothesisGenerator

class LLMHypothesisGenerator(HypothesisGenerator):
    """ LLM-based hypothesis generation """
    def __init__(self, data: str, scoring: str):
        self.keep_cols = [
            "giv1", "siv1", "sivv1", "giv2", "siv2", "sivv2",
            'reg_qualifier', 'num_qualifier', 'mod_qualifier',
            'mod', 'type_mod', 'cat_qualifier']
        self.h = pd.read_csv(data, index_col=0)
        self.h = self.h[~self.h.k.isna()]
        self.h = self.h[[col for col in self.h.columns if col in self.keep_cols]]
        self.h = self.h.to_dict(orient='records')

        self.scoring_o = ["frequency", "entropy", "random"]
        if scoring not in self.scoring_o:
            raise ValueError(f"The `scoring` parameter should be in {self.scoring_o}")
        self.scoring = scoring

    def score_hypotheses(self, hypotheses, top_k):
        """ Scoring hypothesis:
        - random: no scoring
        - frequency: based on the nb of studies for each hypothesis 
        - entropy: entropy score """
        if not hypotheses:
            return None
        # Scores based on the scoring metrics
        if self.scoring_o == "frequency":
            return sorted(hypotheses, key=lambda x: x["k"], reverse=True)[:top_k+1]
        if self.scoring_o == "entropy":
            tot = sum(x['k'] for x in hypotheses)
            scores = [-1*(x['k']/tot)*np.log(x['k']/tot) for x in hypotheses]
            indexes = np.argsort(scores)[::-1][:top_k+1]
            return [hypotheses[i] for i in indexes]
        # self.scoring_o == "random"
        indexes = random.sample(list(range(len(hypotheses))), top_k)
        return [hypotheses[i] for i in indexes]


    def __call__(self, giv: Union[str, None] = None, top_k: int = 1):
        if giv:
            hypotheses = [h for h in self.h if h.get('giv1') == giv]
        else:
            hypotheses = self.h
        top_k = min(top_k, len(hypotheses))
        return self.score_hypotheses(hypotheses=hypotheses, top_k=top_k)


if __name__ == '__main__':
    DATA = "./data/prompt_data_based.csv"
    HG = LLMHypothesisGenerator(data=DATA, scoring="random")
    print("Random: ", HG())
    print("Random + gender: ", HG(giv="gender"))
    print("Random + personality: ", HG(giv="personality"))
    print("==========")
    HG = LLMHypothesisGenerator(data=DATA, scoring="frequency")
    print("Frequency: ", HG())
    print("Frequency + gender: ", HG(giv="gender"))
    print("Frequency + personality: ", HG(giv="personality"))
    print("==========")
    HG = LLMHypothesisGenerator(data=DATA, scoring="entropy")
    print("Entropy: ", HG())
    print("Entropy + gender: ", HG(giv="gender"))
    print("Entropy + personality: ", HG(giv="personality"))
    print("==========")
