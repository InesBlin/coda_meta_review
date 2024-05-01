# -*- coding: utf-8 -*-
""" Expert's recommendation """
import os
import json
import subprocess
import pandas as pd
from src.meta_review import MetaReview, load_json_file
from src.hg.literature import LiteratureHypothesisGenerator
from src.hg.llm import LLMHypothesisGenerator
import streamlit as st
from datetime import datetime

def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as openfile:
        json.dump(data, openfile, indent=4)


def main():
    """ Main """
    st.title("Select a hypothesis")
    st.write("#")
    st.write("We will help you pick a hypothesis to explore with the CoDa databank.")

    with st.form("hypotheses_params_form"):
        hypotheses_selector = st.radio(
            "How would you like to select your hypothesis?",
            ["From literature", "LLM-Based", "LP-Based"],
            index=None, key="hypotheses_selector"
        )
        top_k = st.number_input('How many hypotheses would you like to generate?', value=3, key="top_k")
        submit_hp = st.form_submit_button("Generate Hypotheses Params")

    if submit_hp:
        st.session_state["submit_hp"] = True

    if st.session_state.get("submit_hp") and not st.session_state.get("hypotheses"):
        if st.session_state.hypotheses_selector == "From literature":
            giv = None
            hg = LiteratureHypothesisGenerator()
            hypotheses = hg(giv=giv, top_k=int(top_k))
            st.session_state["hypotheses"] = hypotheses
        if hypotheses_selector == "LLM-Based":
            giv = None
            data = "./data/prompt_data_based.csv"
            scoring_selector = st.radio(
                "How would you like to rank your hypotheses?",
                ["random", "frequency", "entropy"],
                index=None)
            if scoring_selector:
                hg = LLMHypothesisGenerator(data=data, scoring=scoring_selector)
                hypotheses = hg(giv=giv, top_k=int(top_k))
                st.session_state["hypotheses"] = hypotheses

        if hypotheses_selector == "LP-Based":
            st.warning("Not yet implemented")

    if st.session_state.get("hypotheses"):
        with st.form("hypotheses_form"):
            hypothesis_selector = st.radio(
                "Which hypothesis would you like to explore?",
                st.session_state.get("hypotheses"),
                index=None, key="hypothesis_selector"
            )
            submit_h = st.form_submit_button("Submit Hypothesis")
        
        if submit_h:
            st.session_state["submit_h"] = True

    if st.session_state.get("submit_h") and st.session_state.get("hypothesis_selector"):
        folder_name = str(datetime.now())
        folder_name = folder_name[:10]+ "_" + folder_name[11:19]
        folder_name = os.path.join("app", "meta_review", folder_name)
        os.makedirs(folder_name)
        config = load_json_file(filename="./src/configs/meta_review_base.json")
        config.update({"hypothesis": st.session_state.get("hypothesis_selector")})
        json_path = os.path.join(folder_name, "config.json")
        save_json(json_path, config)
        subprocess.call(f"python src/meta_review.py {json_path} {folder_name}", shell=True)
        st.success("The meta-review was successfully generated")
        


if __name__ == "__main__":
    main()