# -*- coding: utf-8 -*-
""" Expert's recommendation """
import os
import pandas as pd
import streamlit as st
from src.hg.literature import LiteratureHypothesisGenerator
from src.knowledge import generate_hypothesis
from src.settings import ROOT_PATH
from src.helpers.interface import display_sidebar

# Only keeping comparison where there is at least three studies
DATA_T = pd.read_csv(
    os.path.join(ROOT_PATH, "data/same_gv_different_and_same_gv_treat_1_2.csv"), index_col=0)
DATA_T = DATA_T[DATA_T.nb > 3]

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
    for k in ["hypotheses", "hypotheses_choice"]:
        if k not in st.session_state:
            st.session_state[k] = []
    for k in ["submit_hp", "submit_fhsh", "hs_diy_vs_assistant"]:
        if k not in st.session_state:
            st.session_state[k] = False


def get_readable_h(h):
    """ Human-readable hypotheses to select (instead of backend, dictionary version) """
    return f"Comparing studies where {h['siv1']} is {h['sivv1']} and studies where " + \
        f"{h['siv2']} is {h['sivv2']}"


def main():
    """ Main """
    init_t1_t2_var()
    st.title("Select a hypothesis")
    st.write("#")

    # For now: only one option = select your hypothesis
    st.session_state["hs_diy_vs_assistant"] = "Select my own hypotheses"

    if st.session_state["hs_diy_vs_assistant"] == "Select my own hypotheses":
        st.write("Please choose parameters for Treatment 1")

        # GIV1
        data = DATA_T
        giv1 = st.selectbox(
                "GIV1", st.session_state["data_t1_t2"].generic1.unique(),
                index=None)
        # SIV1
        if giv1:
            st.session_state["giv1"] = giv1
            data = data[data.generic1 == st.session_state["giv1"]]
            siv1 = st.selectbox(
                    "SIV1", data.siv1.unique(),
                    index=None)
            # SIVV1
            if siv1:
                st.session_state["siv1"] = siv1
                data = data[data.siv1 == st.session_state["siv1"]]
                sivv1 = st.selectbox(
                    "SIVV1", data.sivv1.unique(),
                    index=None)
                
                # GIV2
                if sivv1:
                    st.session_state["sivv1"] = sivv1
                    data = data[data.sivv1 == st.session_state["sivv1"]]
                    giv2 = st.selectbox(
                        "GIV2", data.generic2.unique(),
                        index=None)

                    # SIV2
                    if giv2:
                        st.session_state["giv2"] = giv2
                        data = data[data.generic2 == st.session_state["giv2"]]
                        siv2 = st.selectbox(
                                "SIV2", data.siv2.unique(),
                                index=None)

                        # SIVV2
                        if siv2:
                            st.session_state["siv2"] = siv2
                            data = data[data.siv2 == st.session_state["siv2"]]
                            sivv2 = st.selectbox(
                                    "SIVV2", data.sivv2.unique(),
                                    index=None)

                            # Comparative
                            if sivv2:
                                st.session_state["sivv2"] = sivv2
                                data = data[data.sivv2 == st.session_state["sivv2"]]
                                comparative = st.selectbox(
                                        "Choose your comparative (treatment 1 vs. treatment 2)", ["higher", "lower"],
                                        index=None)
                                
                                # Submit
                                if comparative:
                                    st.session_state["comparative"] = comparative

        # Wrap up the hypothesis
        st.write("---")
        if st.session_state.get("comparative"):
            hypothesis = {
                "giv1": st.session_state["giv1"], "siv1": st.session_state["siv1"],
                "sivv1": st.session_state["sivv1"], "giv2": st.session_state["giv2"],
                "siv2": st.session_state["siv2"], "sivv2": st.session_state["sivv2"],
                "comparative": st.session_state["comparative"]
            }
            st.markdown(f"""
            You have chosen the following hypothesis:
            "{generate_hypothesis(h_dict=hypothesis)}"
            """)

            if st.button("Add to hypotheses"):
                if hypothesis not in st.session_state["hypotheses"]:
                    st.session_state["hypotheses"].append(hypothesis)
                st.session_state["submit_h"] = True
                st.success("Hypothesis added to hypotheses for meta-review", icon="🔥")


    if st.session_state["hs_diy_vs_assistant"] == "Please help me find some interesting hypotheses":

        st.write("We will help you pick a hypothesis to explore with the CoDa databank.")

        with st.form("hypotheses_params_form"):
            hypotheses_selector = st.radio(
                "How would you like to select your hypothesis?",
                # ["From literature", "LLM-Based", "LP-Based"],
                ["From literature"],
                index=None
            )
            top_k = st.number_input('How many hypotheses would you like to generate?',
                                    value=1, key="top_k")
            if st.form_submit_button("Generate Hypotheses Params"):
                st.session_state["submit_hp"] = True
                st.session_state["hypotheses_selector"] = hypotheses_selector

        if st.session_state.get("submit_hp"):
            if not st.session_state.get("hypotheses_choice"):
                if st.session_state.hypotheses_selector == "From literature":
                    giv = None
                    hg = LiteratureHypothesisGenerator()
                    hypotheses = hg(giv=giv, top_k=int(top_k))
                    st.session_state["hypotheses_choice"] = hypotheses
                # if st.session_state.hypotheses_selector == "LLM-Based":
                #     giv = None
            #     data = os.path.join(ROOT_PATH, "data/prompt_data_based.csv")
                #     scoring_selector = st.radio(
                #         "How would you like to rank your hypotheses?",
                #         ["random", "frequency", "entropy"],
                #         index=None)
                #     if scoring_selector:
                #         hg = LLMHypothesisGenerator(data=data, scoring=scoring_selector)
                #         hypotheses = hg(giv=giv, top_k=int(top_k))
                #         st.session_state["hypotheses_choice"] = hypotheses

                # if st.session_state.hypotheses_selector == "LP-Based":
                #     st.warning("Not yet implemented")

            if st.session_state.get("hypotheses_choice"):
                with st.form("hypotheses_form"):
                    readable_to_dict = {get_readable_h(h): h for h in \
                        st.session_state.get("hypotheses_choice")}
                    hypothesis_selector = st.radio(
                        "Which hypothesis would you like to explore?",
                        readable_to_dict.keys(),
                        index=None, key="hypothesis_selector"
                    )
                    if st.form_submit_button("Submit Hypothesis"):
                        st.session_state["submit_h"] = True
                        if hypothesis_selector not in st.session_state["hypotheses"]:
                            st.session_state["hypotheses"].append(
                                readable_to_dict[hypothesis_selector])

    display_sidebar()


if __name__ == "__main__":
    main()
