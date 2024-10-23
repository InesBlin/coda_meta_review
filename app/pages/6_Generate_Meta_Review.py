# -*- coding: utf-8 -*-
""" Generate the meta-review """
import os
import json
import subprocess
from datetime import datetime
import streamlit as st
from src.meta_review import load_json_file
from src.settings import ROOT_PATH
from src.helpers.interface import display_sidebar


def save_json(filepath, data):
    """ self explanatory """
    with open(filepath, "w", encoding="utf-8") as openfile:
        json.dump(data, openfile, indent=4)


def build_config(hypothesis):
    """ Build config file to generate meta-analysis """
    folder_name = str(datetime.now())
    folder_name = folder_name[:10]+ "_" + folder_name[11:19].replace(":", "-")
    folder_name = os.path.join(ROOT_PATH, "app", "meta_review", folder_name)
    os.makedirs(folder_name)
    config = load_json_file(filename=os.path.join(ROOT_PATH, "src/configs/meta_review_base.json"))
    for k in ["template_folder", "label_des", "references", "data"]:
        config[k] = os.path.join(ROOT_PATH, config[k])
    for k, v in config["cached"].items():
        config["cached"][k] = os.path.join(ROOT_PATH, v)
    # Update hypothesis
    config.update({"hypothesis": hypothesis})

    # Update inclusion criteria
    if st.session_state.inclusion_criteria:
        config.update({"inclusion_criteria": st.session_state["inclusion_criteria"]})
    else:
        config.update({"inclusion_criteria": {}})

    # Update variables
    if st.session_state.mr_variables:
        config.update({"control_variables": st.session_state["mr_variables"]})
    else:
        config.update({"control_variables": []})

    # Update analytic strategy
    for as_key in ["type_rma", "es_measure", "method_mv"]:
        config.update({as_key: st.session_state[as_key]})

    # Update custom text
    if st.session_state.get("custom_content"):
        config.update(dict(st.session_state.custom_content.items()))
    json_path = os.path.join(folder_name, "config.json")
    save_json(json_path, config)
    return folder_name


def check_meta_review():
    """ Check various cached parameters to see if the 
    meta-review can be generated
    - warning -> mandatory for the MR
    - info -> informative, can be added """
    if not st.session_state["hypotheses"]:
        st.error("You need to choose at least one hypothesis. " + \
            "Please refer to the 'Select a hypothesis' Section.", icon="🚨")
    for (sskey, val, section) in [
        ("inclusion_criteria", "inclusion criteria", "Inclusion Criteria"),
        ("mr_variables", "control variables", "Variables")
    ]:
        if not st.session_state[sskey]:
            st.info(f"You haven't added any {val}. " + \
                f"If you want to add some, please refer to the '{section}' Section.")
    if not (st.session_state["method_mv"] and \
        st.session_state["es_measure"] and \
            st.session_state["type_rma"]):
        st.error("You need to choose an analytic strategy. " + \
            "Please refer to the 'Analytic Strategy' Section.", icon="🚨")
    if not st.session_state["submit_custom_content"]:
        st.info("You haven't added any custom content. " + \
            "If you want to add some (at least a title+authors), " + \
                "please refer to the 'Custom Text' Section.")

def main():
    """ Main """
    for k in ["submit_custom_content", "inclusion_criteria",
              "mr_variables", "method_mv", "es_measure", "type_rma"]:
        if k not in st.session_state:
            st.session_state[k] = False
    for k in ["hypotheses"]:
        if k not in st.session_state:
            st.session_state[k] = []
    st.title("Generate Meta Review")
    st.write("#")

    check_meta_review()
    if st.session_state.hypotheses:
        hypothesis = st.session_state.hypotheses[0]
        folder_name = build_config(hypothesis=hypothesis)
        json_path = os.path.join(folder_name, "config.json")

        if st.button("Generate Meta-Review"):
            subprocess.call(f"python src/meta_review.py {json_path} {folder_name}", shell=True)
            if os.path.exists(os.path.join(folder_name, 'report.html')):
                st.success("The meta-review was successfully generated")
            else:
                st.error("It looks like the meta-review was not generated, please change hypothesis.")

    display_sidebar()


if __name__ == "__main__":
    main()
