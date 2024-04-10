# -*- coding: utf-8 -*-
"""
Prompting ChatGPT to formulate hypothesis on how humans cooperate
"""
from tqdm import tqdm
from src.pipeline import Pipeline
from kglab.helpers.data_load import read_csv

DATA = read_csv("./data/observationData.csv")
CACHED = {
        "study_moderators": "./data/moderators/study_moderators.csv",
        "country_moderators": "./data/moderators/country_moderators.csv",
        "simple_country_moderators": "./data/moderators/simple_country_moderators.csv",
        "complex_country_moderators": "./data/moderators/complex_country_moderators.csv",
        "variable_moderators": "./data/moderators/variable_moderators.csv"
    }

PARAMS_1 =  {
    "reciprocity": [
        {"giv1": "personality", "siv1": "individual difference",
        "sivv1": "positive reciprocity", "giv2": "personality",
        "siv2": "individual difference", "sivv2": "individualism",},
        {"giv1": "personality", "siv1": "individual difference",
         "sivv1": "positive reciprocity", "giv2": "personality",
         "siv2": "individual difference", "sivv2": "individualism",},
         {"giv1": "emotions", "siv1": "emotion valence",
         "sivv1": "positive", "giv2": "emotions",
         "siv2": "partner's emotion valence", "sivv2": "positive",},
        {"giv1": "emotions", "siv1": "emotion valence",
         "sivv1": "negative", "giv2": "emotions",
         "siv2": "partner's emotion valence", "sivv2": "negative",},
        {"giv1": "leadership", "siv1": "leader's behavior",
         "sivv1": "cooperative", "giv2": "leadership",
         "siv2": "leader's behavior", "sivv2": "noncooperative",},
        {"giv1": "partner(s)'_strategies", "siv1": "iterated pre-programmed cooperation rate level",
         "sivv1": "high", "giv2": "partner(s)'_strategies",
         "siv2": "iterated pre-programmed cooperation rate level", "sivv2": "low",},
        {"giv1": "partner(s)'_strategies", "siv1": "iterated strategy",
         "sivv1": "grim", "giv2": "partner(s)'_strategies",
         "siv2": "iterated strategy", "sivv2": "real partner",},
        {"giv1": "partner(s)'_strategies", "siv1": "iterated strategy",
         "sivv1": "grim", "giv2": "partner(s)'_strategies",
         "siv2": "iterated strategy", "sivv2": "real partner",},
        {"giv1": "partner(s)'_strategies", "siv1": "iterated strategy",
         "sivv1": "tit-for-tat", "giv2": "partner(s)'_strategies",
         "siv2": "iterated strategy", "sivv2": "real partner",},
        {"giv1": "partner(s)'_strategies", "siv1": "iterated strategy",
         "sivv1": "pavlov", "giv2": "partner(s)'_strategies",
         "siv2": "iterated strategy", "sivv2": "real partner",},
        {"giv1": "reputation", "siv1": "knowledge of partner's prior behavior",
         "sivv1": "cooperative", "giv2": "reputation",
         "siv2": "knowledge of partner's prior behavior", "sivv2": "noncooperative",},
    ],
    "social norms": [
        {"giv1": "personality", "siv1": "individual difference",
         "sivv1": "collectivism", "giv2": "personality",
         "siv2": "individual difference", "sivv2": "individualism",},
    ],
    "evolutionary": [
        {"giv1": "personality", "siv1": "individual difference level",
         "sivv1": "low", "giv2": "personality",
         "siv2": "individual difference level", "sivv2": "high",},
        {"giv1": "normative_Behavior", "siv1": "group variability",
         "sivv1": "low", "giv2": "normative_Behavior",
         "siv2": "group variability", "sivv2": "high",},

    ],
    "network": [
        {"giv1": "personality", "siv1": "individual difference level",
         "sivv1": "low", "giv2": "personality",
         "siv2": "individual difference level", "sivv2": "high",},
        {"giv1": "perception_of_the_partner(s)", "siv1": "similarity level",
         "sivv1": "high", "giv2": "perception_of_the_partner(s)",
         "siv2": "similarity level", "sivv2": "low",},
        {"giv1": "identification", "siv1": "partner's group membership",
         "sivv1": "ingroup", "giv2": "identification",
         "siv2": "partner's group membership", "sivv2": "output",},
        {"giv1": "identification", "siv1": "group type",
         "sivv1": "natural group", "giv2": "identification",
         "siv2": "group type", "sivv2": "experimentally induced group",},

    ],
    "institutional": [
        {"giv1": "punishment", "siv1": "punishment rule",
         "sivv1": "contribution-based", "giv2": "punishment",
         "siv2": "punishment rule", "sivv2": "none",},
        {"giv1": "punishment", "siv1": "punishment rule",
         "sivv1": "outcome-based", "giv2": "punishment",
         "siv2": "punishment rule", "sivv2": "none",},
        {"giv1": "punishment", "siv1": "punishment agent",
         "sivv1": "institution", "giv2": "punishment",
         "siv2": "punishment agent", "sivv2": "computer",},
        {"giv1": "reward", "siv1": "reward agent",
         "sivv1": "institution", "giv2": "reward",
         "siv2": "reward agent", "sivv2": "computer",},
        {"giv1": "reward", "siv1": "reward rule",
         "sivv1": "contribution-based", "giv2": "reward",
         "siv2": "reward rule", "sivv2": "random",},
        {"giv1": "reward", "siv1": "sequential reward",
         "sivv1": "sequential turn-taking", "giv2": "reward",
         "siv2": "sequential reward", "sivv2": "simultaneous",},

    ],
}

PARAMS_2 = {
    "reciprocity": [
        {"giv1": "emotions", "siv1": "emotion valence", "sivv1": "positive",
         "giv2": "emotions", "siv2": "emotion valence", "sivv2": "negative",},
        {"giv1": None, "siv1": None, "sivv1": None,
         "giv2": None, "siv2": None, "sivv2": None,},
    ],
    "social norms": [
        {"giv1": "leadership", "siv1": "leader's behavior", "sivv1": "cooperative",
         "giv2": "leadership", "siv2": "leader's behavior", "sivv2": "noncooperative",},
        {"giv1": "personality", "siv1": "individual difference level", "sivv1": "low",
         "giv2": "personality", "siv2": "individual difference level", "sivv2": "medium",},
        {"giv1": "punishment", "siv1": "sequential punishment", "sivv1": "sequential turn-taking",
         "giv2": "punishment", "siv2": "sequential punishment", "sivv2": "simultaneous",},
        {"giv1": None, "siv1": None, "sivv1": None,
         "giv2": None, "siv2": None, "sivv2": None,},
    ],
    "evolutionary": [
        {"giv1": "group_Size", "siv1": "decision maker", "sivv1": "group",
         "giv2": "group_Size", "siv2": "decision maker", "sivv2": "individual",},
    ],
    "network": [
        {"giv1": "acquaintance", "siv1": "relationship with the partner", "sivv1": "acquaintance",
         "giv2": "acquaintance", "siv2": "relationship with the partner", "sivv2": "stranger",},
        {"giv1": "acquaintance", "siv1": "relationship with the partner", "sivv1": "friend",
         "giv2": "acquaintance", "siv2": "relationship with the partner", "sivv2": "stranger",},
        {"giv1": "identification", "siv1": "group type", "sivv1": "natural group",
         "giv2": "identification", "siv2": "partner's group membership", "sivv2": "outgroup",},
        {"giv1": None, "siv1": None, "sivv1": None,
         "giv2": None, "siv2": None, "sivv2": None,},
    ],
    "institutional": [
        {"giv1": "punishment", "siv1": "punishment rule", "sivv1": "rank-based",
         "giv2": "punishment", "siv2": "punishment treatment", "sivv2": "-1",},
        {"giv1": "reward", "siv1": "reward rule", "sivv1": "contribution-based",
         "giv2": "reward", "siv2": "reward treatment", "sivv2": "1",},
        {"giv1": None, "siv1": None, "sivv1": None,
         "giv2": None, "siv2": None, "sivv2": None,},
    ]
}

PARAMS_ONTOLOGY_BASED = {
    "punishment": [
        {"giv1": "punishment", "siv1": "punishment rule", "sivv1": "outcome-based",
         "giv2": "punishment", "siv2": "punishment rule", "sivv2": 'other',},
        {"giv1": "punishment", "siv1": "punishment sequential", "sivv1": "sequential leadership-by-example",
         "giv2": "punishment", "siv2": "punishment sequential", "sivv2": "simultaneous",},
         {"giv1": "punishment", "siv1": "punishment agent", "sivv1": "institution",
         "giv2": "punishment", "siv2": "punishment agent", "sivv2": "computer",},
         {"giv1": "punishment", "siv1": "punishment distribution rule", "sivv1": "deductive",
         "giv2": "punishment", "siv2": "punishment distribution rule", "sivv2": "redistributive",},
        {"giv1": "punishment", "siv1": "punishment incentive", "sivv1": "hypothetical",
         "giv2": "punishment", "siv2": "punishment incentive", "sivv2": "non-monetary social",},
    ],
    "personality": [
        {"giv1": "personality", "siv1": "altruism", "sivv1": "high",
         "giv2": "personality", "siv2": "altruism", "sivv2": 'medium',},
        {"giv1": "personality", "siv1": "machiavellianism", "sivv1": "low",
         "giv2": "personality", "siv2": "machiavellianism", "sivv2": 'high',},
        {"giv1": "personality", "siv1": "emotional intelligence", "sivv1": "low",
         "giv2": "personality", "siv2": "emotional intelligence", "sivv2": 'high',},
        {"giv1": "personality", "siv1": "trust propensity", "sivv1": "high",
         "giv2": "personality", "siv2": "trust propensity", "sivv2": 'low',},
        {"giv1": "personality", "siv1": "conscientiousness", "sivv1": "low",
         "giv2": "personality", "siv2": "conscientiousness", "sivv2": 'high',},
    ],
    "communication": [
        {"giv1": "communication", "siv1": "communication occurrence", "sivv1": "one-shot",
         "giv2": "communication", "siv2": "communication occurrence", "sivv2": 'ongoing',},
        {"giv1": "communication", "siv1": "communication content", "sivv1": "threat",
         "giv2": "communication", "siv2": "communication content", "sivv2": 'request',},
        {"giv1": "communication", "siv1": "communication type", "sivv1": "written",
         "giv2": "communication", "siv2": "communication type", "sivv2": 'verbal',},
        {"giv1": "communication", "siv1": "communication content", "sivv1": "irrelevant",
         "giv2": "communication", "siv2": "communication content", "sivv2": 'other',},
        {"giv1": "communication", "siv1": "communication type", "sivv1": "nonverbal",
         "giv2": "communication", "siv2": "communication type", "sivv2": 'verbal',},
    ],
    "reward": [
        {"giv1": "reward", "siv1": "reward agent", "sivv1": "comptuter",
         "giv2": "reward", "siv2": "reward agent", "sivv2": 'peer',},
        {"giv1": "reward", "siv1": "reward sequential", "sivv1": "sequential leadership-by-example",
         "giv2": "reward", "siv2": "reward sequential", "sivv2": "simultaneous",},
        {"giv1": "reward", "siv1": "reward incentive", "sivv1": "monetary",
         "giv2": "reward", "siv2": "reward incentive", "sivv2": "non-monetary material",},
         {"giv1": "reward", "siv1": "reward rule", "sivv1": "outcome-based",
         "giv2": "reward", "siv2": "reward rule", "sivv2": "rank-based",},
         {"giv1": "reward", "siv1": "reward treatment", "sivv1": "-1",
         "giv2": "reward", "siv2": "reward treatment", "sivv2": "0",},
    ],
    "identification": [
        {"giv1": "identification", "siv1": "identification level", "sivv1": "medium",
         "giv2": "identification", "siv2": "identification level", "sivv2": 'low',},
        {"giv1": "identification", "siv1": "group type", "sivv1": "natural group",
         "giv2": "identification", "siv2": "group type", "sivv2": 'experimentally induced group',},
        {"giv1": "identification", "siv1": "target membership", "sivv1": "stranger",
         "giv2": "identification", "siv2": "target membership", "sivv2": 'outgroup',},
        {"giv1": "identification", "siv1": "knowledge of membership", "sivv1": "common",
         "giv2": "identification", "siv2": "knowledge of membership", "sivv2": 'unknown',},
        {"giv1": "identification", "siv1": "entitativity level", "sivv1": "high",
         "giv2": "identification", "siv2": "entitativity level", "sivv2": 'medium',},
    ]
}

# {"giv1": "", "siv1": "", "sivv1": "",
#  "giv2": "", "siv2": "", "sivv2": '',},

def print_params():
    """ Print params to transfer to excel """
    for k, info in PARAMS_ONTOLOGY_BASED.items():
        for p in info:
            print(f"{k},{p['giv1']},{p['siv1']},{p['sivv1']},{p['giv2']},{p['siv2']},{p['sivv2']}")

def get_param_description(k, p):
    return f"{k},{p['giv1']},{p['siv1']},{p['sivv1']},{p['giv2']},{p['siv2']},{p['sivv2']}"

def run_meta_analysis():
    """ Print params to transfer to excel """
    final_info = []
    for k, info in tqdm(PARAMS_ONTOLOGY_BASED.items()):
        for p in info:
            if p.get("giv1"):
                try:
                    pipeline = Pipeline(
                        giv1=p['giv1'], siv1=p['siv1'], sivv1=p['sivv1'],
                        giv2=p['giv2'], siv2=p['siv2'], sivv2=p['sivv2'], **CACHED)
                    output = pipeline(data=DATA, mods=None)
                    curr_res = output["results_rma"]
                    metrics = ",".join([str(curr_res[x].reshape((1,))[0]) for x in ["k", "b", "pval"]])
                    final_info.append(get_param_description(k=k, p=p) + "," + metrics)
                except Exception as e:
                    print(e)
                    final_info.append(get_param_description(k=k, p=p) + "," + "N/A,N/A,N/A")
    return final_info


if __name__ == '__main__':
    FINAL_INFO = run_meta_analysis()
    for line in FINAL_INFO:
        print(line)
