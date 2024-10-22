# -*- coding: utf-8 -*-
""" Display the meta-review """
import os
import streamlit as st
import streamlit.components.v1 as components

ES_MEASURE_TO_K = {
    "Cohen's standardized mean difference (d)": "d",
    "Pearson's correlation coefficient (r)": "r"}
K_TO_ES_MEASURE = {v: k for k, v in ES_MEASURE_TO_K.items()}

def get_source_code(html_path: str) -> str:
    """ Return graph visualisation HTML """
    with open(html_path, 'r', encoding='utf-8') as html_file:
        source_code = html_file.read()
    return source_code

def main():
    """ Main """
    if "submit_h" not in st.session_state:
        st.session_state["submit_h"] = False
    mrs = sorted([x for x in os.listdir("app/meta_review") if x[0].isdigit()])
    mrs = [x for x in mrs if "report.html" in os.listdir(os.path.join("app/meta_review", x))]
    if mrs and st.session_state.submit_h:
        path = os.path.join("app/meta_review", mrs[-1], "report.html")
        source_code = get_source_code(html_path=path)
        components.html(source_code, scrolling=False, height=10000)
    else:
        st.warning("It looks like your meta-review was not generated.")
    
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
        if st.session_state.get("submit_as") and st.session_state["method_mv"] \
        and st.session_state["es_measure"]:
            st.write("You have chosen the following analytic strategy:")
            st.write(f'Effect size {K_TO_ES_MEASURE[st.session_state["es_measure"]]} with {st.session_state["method_mv"]} model')
        if st.session_state.get("custom_content"):
            st.write("You have added the following custom content:")
            st.write({k: v for k, v in st.session_state["custom_content"].items() if v})


if __name__ == '__main__':
    main()