# -*- coding: utf-8 -*-
""" Generate the meta-review """
import os
import json
import subprocess
from datetime import datetime
import streamlit as st
from src.meta_review import load_json_file

def save_json(filepath, data):
    """ self explanatory """
    with open(filepath, "w", encoding="utf-8") as openfile:
        json.dump(data, openfile, indent=4)

def build_config(hypothesis):
    """ Build config file to generate meta-analysis """
    folder_name = str(datetime.now())
    folder_name = folder_name[:10]+ "_" + folder_name[11:19]
    folder_name = os.path.join("app", "meta_review", folder_name)
    os.makedirs(folder_name)
    config = load_json_file(filename="./src/configs/meta_review_base.json")
    # Update hypothesis
    config.update({"hypothesis": hypothesis})

    # Update inclusion criteria
    if st.session_state.inclusion_criteria:
        config.update({"inclusion_criteria": st.session_state["inclusion_criteria"]})
    else:
        config.update({"inclusion_criteria": {}})

    # Update variables
    if st.session_state.mr_variables:
        config.update({"coding_variables": st.session_state["mr_variables"]})
    else:
        config.update({"coding_variables": []})

    # Update analytic strategy
    for as_key in ["type_rma", "es_measure", "method_mv"]:
        config.update({as_key: st.session_state[as_key]})

    # Update custom text
    if st.session_state.get("custom_content"):
        config.update(dict(st.session_state.custom_content.items()))
    json_path = os.path.join(folder_name, "config.json")
    save_json(json_path, config)
    return folder_name

def main():
    """ Main """
    for k in ["hypotheses", "submit_custom_content"]:
        if k not in st.session_state:
            st.session_state[k] = False
    st.title("Generate Meta Review")
    st.write("#")

    if not st.session_state["hypotheses"]:
        st.error("You need to choose at least one hypotheses. Please refer to page `Select a hypothesis`.")
    if not st.session_state["submit_custom_content"]:
        st.warning("You haven't updated any custom content. Please do so in the `Custom Text` section.", icon="🚨")


    if st.session_state.hypotheses:
        hypothesis = st.session_state.hypotheses[0]
        folder_name = build_config(hypothesis=hypothesis)
        json_path = os.path.join(folder_name, "config.json")

        if st.button("Generate Meta-Review"):
            subprocess.call(f"python src/meta_review.py {json_path} {folder_name}", shell=True)
            st.success("The meta-review was successfully generated")


if __name__ == "__main__":
    main()