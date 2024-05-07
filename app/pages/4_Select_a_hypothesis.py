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

DATA_T = pd.read_csv("./data/same_gv_different_and_same_gv_treat_1_2.csv", index_col=0)
DATA_T = DATA_T[DATA_T.nb > 0]

def reinit_t1_t2_var():
    """ Init treatment 1 vs. treatment 2 session state variables """
    for var in ["giv1", "siv1", "sivv1", "giv2", "siv2", "sivv2"]:
        st.session_state[var] = None
        st.session_state[f"submit_hs_dva_f_{var}"] = False
    st.session_state["data_t1_t2"] = DATA_T

def init_t1_t2_var():
    """ Init treatment 1 vs. treatment 2 session state variables 
    (if not already in the session state) """
    for var in ["giv1", "siv1", "sivv1", "giv2", "siv2", "sivv2"]:
        if var not in st.session_state:
            st.session_state[var] = None
        if f"submit_hs_dva_f_{var}" not in st.session_state:
            st.session_state[f"submit_hs_dva_f_{var}"] = False
    if "data_t1_t2" not in st.session_state:
        st.session_state["data_t1_t2"] = DATA_T

def update_val(**kwargs):
    """ Update (key, value) pair in session state """
    st.session_state[kwargs["key"]] = kwargs["val"]

def save_json(filepath, data):
    """ self explanatory """
    with open(filepath, "w", encoding="utf-8") as openfile:
        json.dump(data, openfile, indent=4)


def main():
    """ Main """
    init_t1_t2_var()
    if "hypotheses" not in st.session_state:
        st.session_state["hypotheses"] = []
    st.title("Select a hypothesis")
    st.write("#")

    hs_diy_vs_assistant = st.radio(
        "How would you like to select your hypothesis?",
        ["Select my own hypotheses", "Please help me find some interesting hypotheses"],
        index=None, key="hs_diy_vs_assistant"
    )

    if hs_diy_vs_assistant == "Select my own hypotheses":
        st.write("Please choose parameters for Treatment 1")

        # GIV1
        with st.form("hs_dva_f_giv1"):
            giv1 = st.selectbox(
                "GIV1", st.session_state["data_t1_t2"].generic1.unique(),
                index=None)
            if st.form_submit_button("Save GIV1"):
                st.session_state["giv1"] = giv1
                st.session_state["submit_hs_dva_f_giv1"] = True

        # SIV1
        if st.session_state.get("giv1") and st.session_state.get("submit_hs_dva_f_giv1"):
            st.session_state["data_t1_t2"] = st.session_state["data_t1_t2"] \
                [st.session_state["data_t1_t2"].generic1 == st.session_state["giv1"]]
            with st.form("hs_dva_f_siv1"):
                siv1 = st.selectbox(
                    "SIV1", st.session_state["data_t1_t2"].siv1.unique(),
                    index=None)
                if st.form_submit_button("Save SIV1"):
                    st.session_state["siv1"] = siv1
                    st.session_state["submit_hs_dva_f_siv1"] = True

        # SIVV1
        if st.session_state.get("siv1") and st.session_state.get("submit_hs_dva_f_siv1"):
            st.session_state["data_t1_t2"] = st.session_state["data_t1_t2"] \
                [st.session_state["data_t1_t2"].siv1 == st.session_state["siv1"]]
            with st.form("hs_dva_f_sivv1"):
                sivv1 = st.selectbox(
                    "SIVV1", st.session_state["data_t1_t2"].sivv1.unique(),
                    index=None)
                if st.form_submit_button("Save SIVV1"):
                    st.session_state["sivv1"] = sivv1
                    st.session_state["submit_hs_dva_f_sivv1"] = True

        # GIV2
        if st.session_state.get("sivv1") and st.session_state.get("submit_hs_dva_f_sivv1"):
            st.session_state["data_t1_t2"] = st.session_state["data_t1_t2"] \
                [st.session_state["data_t1_t2"].sivv1 == st.session_state["sivv1"]]
            with st.form("hs_dva_f_giv2"):
                giv2 = st.selectbox(
                    "GIV2", st.session_state["data_t1_t2"].generic2.unique(),
                    index=None)
                if st.form_submit_button("Save GIV2"):
                    st.session_state["giv2"] = giv2
                    st.session_state["submit_hs_dva_f_giv2"] = True

        # SIV2
        if st.session_state.get("giv2") and st.session_state.get("submit_hs_dva_f_giv2"):
            st.session_state["data_t1_t2"] = st.session_state["data_t1_t2"] \
                [st.session_state["data_t1_t2"].generic2 == st.session_state["giv2"]]
            with st.form("hs_dva_f_siv2"):
                siv2 = st.selectbox(
                    "SIV2", st.session_state["data_t1_t2"].siv2.unique(),
                    index=None)
                if st.form_submit_button("Save SIV2"):
                    st.session_state["siv2"] = siv2
                    st.session_state["submit_hs_dva_f_siv2"] = True

        # SIVV2
        if st.session_state.get("siv2") and st.session_state.get("submit_hs_dva_f_siv2"):
            st.session_state["data_t1_t2"] = st.session_state["data_t1_t2"] \
                [st.session_state["data_t1_t2"].siv2 == st.session_state["siv2"]]
            with st.form("hs_dva_f_sivv2"):
                sivv2 = st.selectbox(
                    "SIVV2", st.session_state["data_t1_t2"].sivv2.unique(),
                    index=None)
                if st.form_submit_button("Save SIVV2"):
                    st.session_state["sivv2"] = sivv2
                    st.session_state["submit_hs_dva_f_sivv2"] = True

        # Wrap up the hypothesis
        if st.session_state.get("sivv2"):
            hypothesis = {
                "giv1": st.session_state["giv1"], "siv1": st.session_state["siv1"],
                "sivv1": st.session_state["sivv1"], "giv2": st.session_state["giv2"],
                "siv2": st.session_state["siv2"], "sivv2": st.session_state["sivv2"]
            }
            st.markdown(f"""You have chosen one hypothesis:\\
            ```
            {hypothesis}
            ```""")

            if st.button("Add to hypotheses"):
                if hypothesis not in st.session_state["hypotheses"]:
                    st.session_state["hypotheses"].append(hypothesis)
                st.success("Hypothesis added to hypotheses for meta-review", icon="🔥")

            if st.button("Choose another hypothesis"):
                reinit_t1_t2_var()
                st.rerun()


    if hs_diy_vs_assistant == "Please help me find some interesting hypotheses":

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

    with st.sidebar:
        st.write("You have chosen the following hypotheses:")
        for h in st.session_state["hypotheses"]:
            st.write(h)


if __name__ == "__main__":
    main()