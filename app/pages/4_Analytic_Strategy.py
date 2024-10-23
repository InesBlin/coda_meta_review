# -*- coding: utf-8 -*-
""" Analytic strategy for the meta-analysis """
import streamlit as st
from src.helpers.interface import ES_MEASURE_TO_K
from src.helpers.interface import display_sidebar

TYPE_RMA_TO_K = {"simple": "uni"}#, "multilevel": "mv"}

METHODS = {
    "uni": ["EE", "DL", "HE", "HS", "HSk", "SJ",
            "ML", "REML", "EB", "PM", "GENQ", "PMM", "GENQM"],
    "mv": ["ML", "REML"]
}

def main():
    """ Main """
    for val in ["submit_type_rma", "submit_as", "submit_as_whole", 
                "submit_mr_variables", "analytic_strategy"]:
        if val not in st.session_state:
            st.session_state[val] = False
    st.title("Analytic Strategy")
    st.write("#")

    st.write("Please choose your analytic strategy for the meta-review ; the model is a simple model.")

    # type rma
    # type_rma = st.selectbox(
    #     "Do you want to use a simple or multilevel model?", 
    #     TYPE_RMA_TO_K.keys(),
    #     index=None)
    type_rma = "simple"
    # es measure
    if type_rma:
        st.session_state["type_rma"] = TYPE_RMA_TO_K[type_rma]
        es_measure = st.selectbox(
            "Which effect size measure do you want to use?", 
            ES_MEASURE_TO_K.keys(),
            index=None)

        methods = METHODS[st.session_state["type_rma"]]
        method_mv = st.selectbox(
            "Which model do you want to use?",
            methods, index=None)

        if es_measure and method_mv:
            if st.button("Save params"):
                st.session_state["submit_as"] = True
                st.session_state["method_mv"] = method_mv
                st.session_state["es_measure"] = ES_MEASURE_TO_K[es_measure]
                st.session_state["analytic_strategy"] = True
                st.success("Analytic strategy saved for the meta-review", icon="🔥")

    display_sidebar()


if __name__ == '__main__':
    main()
