# -*- coding: utf-8 -*-
""" Analytic strategy for the meta-analysis """
import os
import streamlit as st
import streamlit.components.v1 as components

ES_MEASURE_TO_K = {
    "Cohen's standardized mean difference (d)": "d",
    "Pearson's correlation coefficient (r)": "r"}
K_TO_ES_MEASURE = {v: k for k, v in ES_MEASURE_TO_K.items()}

SECTION_NAMES = [
    "introduction_custom", "hypothesis_custom", "methods_custom",
    "inclusion_criteria_custom", "coding_effect_sizes_custom",
    "control_variables_custom", "analytic_strategy_custom",
    "results_custom", "discussion_custom"
]

def get_source_code(html_path: str) -> str:
    """ Return graph visualisation HTML """
    with open(html_path, 'r', encoding='utf-8') as html_file:
        source_code = html_file.read()
    return source_code

def main():
    """ Main """
    for var in ["submit_custom_content", "custom_content"]:
        if var not in st.session_state:
            st.session_state[var] = False

    st.title("Adding Custom Text")
    st.write("#")

    st.write("At the end of this page, you can find the HTML template of " + \
        "the meta-review that will be generated.")
    st.write("If you want, you can add content at the end of each section.")

    with st.form("form_custom_text"):
        custom_text = {}
        custom_text["title"] = st.text_input("Title")
        custom_text["authors"] = st.text_input("Authors")
        for sn in SECTION_NAMES:
            display = "Enter text for " + " ".join([x.capitalize() for x in sn.split("_")[:-1]])
            custom_text[sn] = st.text_area(display)
        if st.form_submit_button("Save custom content"):
            st.session_state["submit_custom_content"] = True
            st.session_state["custom_content"] = custom_text
            st.success("Custom content saved for the meta-review", icon="🔥")
    
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

    path = os.path.join("meta_review/templates", "report_template.html")
    source_code = get_source_code(html_path=path)
    components.html(source_code, scrolling=True, height=5000)


if __name__ == '__main__':
    main()
