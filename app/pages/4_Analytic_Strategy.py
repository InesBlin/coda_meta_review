# -*- coding: utf-8 -*-
""" Analytic strategy for the meta-analysis """
import streamlit as st

TYPE_RMA_TO_K = {"simple": "uni"}#, "multilevel": "mv"}
ES_MEASURE_TO_K = {
    "Cohen's standardized mean difference (d)": "d",
    "Pearson's correlation coefficient (r)": "r"}
K_TO_ES_MEASURE = {v: k for k, v in ES_MEASURE_TO_K.items()}
METHODS = {
    "uni": ["EE", "DL", "HE", "HS", "HSk", "SJ",
            "ML", "REML", "EB", "PM", "GENQ", "PMM", "GENQM"],
    "mv": ["ML", "REML"]
}

def main():
    """ Main """
    for val in ["submit_type_rma", "submit_as", "submit_as_whole", "submit_mr_variables"]:
        if val not in st.session_state:
            st.session_state[val] = False
    st.title("Analytic Strategy")
    st.write("#")

    st.write("Please choose your analytic strategy for the meta-review")

    analytic_strategy = {}

    with st.form("form_type_rma"):
        analytic_strategy["type_rma"] = st.selectbox(
            "Do you want to use a simple or multilevel model?", 
            TYPE_RMA_TO_K.keys(),
            index=None)
        if st.form_submit_button("Save model type"):
            st.session_state["submit_type_rma"] = True
            st.session_state["type_rma"] = TYPE_RMA_TO_K[analytic_strategy["type_rma"]]

    if st.session_state["submit_type_rma"] and st.session_state["type_rma"]:
        with st.form("form_analytic_strategy"):
            analytic_strategy["es_measure"] = st.selectbox(
            "Which effect size measure do you want to use?", 
            ES_MEASURE_TO_K.keys(),
            index=None)

            methods = METHODS[st.session_state["type_rma"]]
            analytic_strategy["method_mv"] = st.selectbox(
                "Which model do you want to use?",
                methods, index=None
            )
            if st.form_submit_button("Save params"):
                st.session_state["submit_as"] = True
                st.session_state["method_mv"] = analytic_strategy["method_mv"]
                st.session_state["es_measure"] = ES_MEASURE_TO_K[analytic_strategy["es_measure"]]

    if st.session_state["submit_as"] and st.session_state["method_mv"] \
        and st.session_state["es_measure"]:
        # if st.button("Save Analytic Strategy"):
            # st.session_state["submit_as_whole"] = True
        st.success("Analytic strategy saved for the meta-review", icon="🔥")
    
    with st.sidebar:
        if st.session_state.get("hypotheses"):
            st.write("You have chosen the following hypotheses:")
            for hypothesis in st.session_state["hypotheses"]:
                st.write(hypothesis)
        if st.session_state.get("inclusion_criteria"):
            st.write("You have chosen the following inclusion criteria:")
            st.write(st.session_state["inclusion_criteria"])
        if st.session_state.get("mr_variables"):
            st.write("You have chosen the following control variables:")
            st.write(st.session_state["mr_variables"])
        if st.session_state["submit_as"] and st.session_state["method_mv"] \
        and st.session_state["es_measure"]:
            st.write("You have chosen the following analytic strategy:")
            st.write(f'Effect size {K_TO_ES_MEASURE[st.session_state["es_measure"]]} with {st.session_state["method_mv"]} model')


if __name__ == '__main__':
    main()
