# -*- coding: utf-8 -*-
""" Interface streamlit helpers """

import streamlit as st

ES_MEASURE_TO_K = {
    "Cohen's standardized mean difference (d)": "d",
    "Pearson's correlation coefficient (r)": "r"}
K_TO_ES_MEASURE = {v: k for k, v in ES_MEASURE_TO_K.items()}

def write_param(key, label):
    """ Write parameter for sidebar """
    st.markdown(f"{label}:")
    if st.session_state.get(key):
        if key == "hypotheses":
            for hypothesis in st.session_state["hypotheses"]:
                st.write(hypothesis)
        elif key == "analytic_strategy":
            st.write(f'Effect size {K_TO_ES_MEASURE[st.session_state["es_measure"]]}' + \
                f' with {st.session_state["method_mv"]} model')
        elif key == "custom_content":
            st.write({k: v for k, v in st.session_state["custom_content"].items() if v})
        else:
            st.write(st.session_state.get(key))
    st.write("---")

def display_sidebar():
    """ Display sidebar with parameters chosen for the meta-analysis """
    with st.sidebar:
        st.markdown("**You have chosen the following parameters for your meta-review:**")
        for k, l in [
            ("hypotheses", "**Hypothesis (mandatory)**"), ("inclusion_criteria", "*Inclusion criteria (optional)*"),
            ("mr_variables", "*Control variables (optional)*"), ("analytic_strategy", "**Analytic strategy (mandatory)**"),
            ("custom_content", "*Custom content (optional)*")
        ]:
            write_param(key=k, label=l)
        # st.write("You have chosen the following hypotheses:")
        # if st.session_state.get("hypotheses"):
        #     for hypothesis in st.session_state["hypotheses"]:
        #         st.write(hypothesis)
        # st.write("---")
        # st.write("You have chosen the following inclusion criteria:")
        # if st.session_state.get("inclusion_criteria"):
        #     st.write(st.session_state["inclusion_criteria"])
        # st.write("You have chosen the following control variables:")
        # if st.session_state.get("mr_variables"):
        #     st.write(st.session_state["mr_variables"])
        # st.write("You have chosen the following analytic strategy:")
        # if st.session_state.get("submit_as") and st.session_state["method_mv"] \
        # and st.session_state["es_measure"]:
        #     st.write(f'Effect size {K_TO_ES_MEASURE[st.session_state["es_measure"]]} with {st.session_state["method_mv"]} model')
        # st.write("You have added the following custom content:")
        # if st.session_state.get("custom_content"):
        #     st.write({k: v for k, v in st.session_state["custom_content"].items() if v})
