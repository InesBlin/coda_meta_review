# -*- coding: utf-8 -*-
""" Expert's recommendation """
import pandas as pd
from src.meta_review import MetaReview
import streamlit as st
from src.hg.literature import LiteratureHypothesisGenerator
from src.hg.llm import LLMHypothesisGenerator


st.title("Select a hypothesis")
st.write("#")
st.write("We will help you pick a hypothesis to explore with the CoDa databank.")
hypothesis_selector = st.radio(
    "How would you like to select your hypothesis?",
    ["From literature", "LLM-Based", "LP-Based"],
    index=None,
)

st.write("You selected:", hypothesis_selector)

top_k = st.number_input('How many hypotheses would you like to generate?', value=3)
hypotheses = None

if hypothesis_selector == "From literature":
    giv = None
    hg = LiteratureHypothesisGenerator()
    hypotheses = hg(giv=giv, top_k=int(top_k))

if hypothesis_selector == "LLM-Based":
    giv = None
    data = "./data/prompt_data_based.csv"
    scoring_selector = st.radio(
        "How would you like to rank your hypotheses?",
        ["random", "frequency", "entropy"],
        index=None)
    if scoring_selector:
        hg = LLMHypothesisGenerator(data=data, scoring=scoring_selector)
        hypotheses = hg(giv=giv, top_k=int(top_k))

if hypothesis_selector == "LP-Based":
    st.warning("Not yet implemented")

if hypotheses:
    hypothesis = st.radio(
        "Which hypothesis would you like to explore?",
        hypotheses,
        index=None)
    if hypothesis:
        st.write(hypothesis)
        # ARGS = {
        #     'template_folder': 'meta_review/templates',
        #     'config': 'meta_review/config.yaml',
        #     'structure': 'meta_review/templates/structure.yaml',
        #     'references': 'meta_review/templates/references.json',
        #     "data": pd.read_csv("./data/observationData.csv", index_col=0),
        #     "cached": {
        #         "study_moderators": "./data/moderators/study_moderators.csv",
        #         "country_moderators": "./data/moderators/country_moderators.csv",
        #         "simple_country_moderators": "./data/moderators/simple_country_moderators.csv",
        #         "complex_country_moderators": "./data/moderators/complex_country_moderators.csv",
        #         "variable_moderators": "./data/moderators/variable_moderators.csv"
        #     },
        #     "report_template": 'report_template.html',
        # }
        # ARGS.update(hypothesis)
        # meta_review = MetaReview(**ARGS)
        # meta_review(save_folder="./app/meta_review")
