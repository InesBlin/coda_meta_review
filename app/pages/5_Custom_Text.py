# -*- coding: utf-8 -*-
""" Analytic strategy for the meta-analysis """
import os
import streamlit as st
import streamlit.components.v1 as components

SECTION_NAMES = [
    "introduction_custom", "hypothesis_custom", "methods_custom",
    "inclusion_criteria_custom", "coding_effect_sizes_custom",
    "coding_variables_custom", "analytic_strategy_custom",
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

    if not (st.session_state["submit_as"] and st.session_state["method_mv"] and st.session_state["es_measure"]):
        st.warning("You haven't chosen your analytic strategy. Please do so in the `Analytic Strategy` section.", icon="🚨")

    st.write("At the end of this page, you can find the HTML template of the meta-review that will be generated.")
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

    path = os.path.join("meta_review/templates", "report_template.html")
    source_code = get_source_code(html_path=path)
    components.html(source_code, scrolling=True, height=5000)


if __name__ == '__main__':
    main()
