# -*- coding: utf-8 -*-
""" Inclusion Criteria """
import os
import streamlit as st
import pandas as pd
from src.inclusion_criteria import InclusionCriteria
from src.settings import ROOT_PATH
from src.data_selection import DataSelector
from src.data_prep import DataPrep

IC = InclusionCriteria()
IC_TYPES = sorted(IC.mappings.keys())
DATA = pd.read_csv(os.path.join(ROOT_PATH, "data/observationData.csv"), index_col=0)

if len(st.session_state["hypotheses"]) == 0:
    st.warning("You haven't chosen a hypothesis yet. To do so, " + \
        "please refer to the page 'Select a hypothesis'.")
else:
    # Filtering data based on hypothesis (to ensure inclusion criteria are consistant with data)
    h = st.session_state["hypotheses"][0]
    data_selector = DataSelector(siv1=h["siv1"], siv2=h["siv2"], sivv1=h["sivv1"], sivv2=h["sivv2"])
    data_prep = DataPrep(siv1=h["siv1"], sivv1=h["sivv1"], siv2=h["siv2"], sivv2=h["sivv2"])
    DATA = data_selector(data=DATA)
    DATA = data_prep(filtered_data=DATA)

DATA["lang"]=DATA["observationName"].apply(lambda x: x[:3])

@st.cache_data
def get_options_simple_ic(data, val, type_ic='simple'):
    """ Unique values of column `val` in `data` """
    if type_ic == 'simple':
        vals = data[~data[val].isna()][val].unique()
        vals = set(y for x in vals for y in x.split(","))
    else:
        vals = data[~data[val].isna()][val].unique()
    return sorted(vals)

@st.cache_data
def get_range_ic(data, val):
    """ Min/Max values """
    vals = data[~data[val].isna()][val].unique()
    if vals.shape[0] > 0:
        correct_vals = sorted([float(val) for val in vals \
            if isinstance(val, (int, float)) or \
                (isinstance(val, str) and val.isdigit())])
        if correct_vals:
            return correct_vals[0], correct_vals[-1]
        return None, None
    return None, None


def main():
    """ Main for inclusion criteria """
    if 'inclusion_criteria' not in st.session_state:
        st.session_state["inclusion_criteria"] = None

    st.title("Inclusion Criteria")

    st.markdown("""
    You can now choose additional inclusion criteria.
    
    Given the hypothesis you have chosen in the previous step, here is the filtered data for the meta-analysis.
    
    Although the inclusion critera enable you to filter your data further, bear in mind that it can also drastically reduce the size of your data.

    Please note that if an inclusion criteria only has one value across the below data, it will not be proposed to you as inclusion criteria.
    """)
    st.write(DATA)
    st.markdown("---")
    type_ic = st.multiselect(
        "Which inclusion criteria would you like to add?",
        IC_TYPES
    )
    params = {k: {} for k in type_ic}
    with st.form(f"ic"):
        for tic in type_ic:
            [_, _, simple_ic, range_ic] = IC.mappings[tic]
        # with st.form(f"ic_{tic}"):
            for simple in simple_ic:
                options = get_options_simple_ic(data=DATA, val=simple)
                if len(options) > 1:
                    params[tic][simple] = st.multiselect(
                        f'Choose your inclusion criteria for `{simple}`',
                        options=get_options_simple_ic(data=DATA, val=simple)
                    )
            for range_ in range_ic:
                min_range, max_range = get_range_ic(data=DATA, val=range_)
                if min_range and (min_range != max_range):
                    params[tic][range_] = st.slider(
                        f"Select your range of values for {range_}", 
                        min_range, max_range, (min_range, max_range),
                        key=f'{tic}_{range_}'
                    )
            # if st.form_submit_button(f"Save {tic} inclusion criteria"):
            #     st.session_state[f"submit_ic_{tic}"] = True
        
        if st.form_submit_button(f"Save inclusion criteria"):
                st.session_state[f"submit_ic"] = True

    params_non_null = {}
    for k1, v1 in params.items():
        params_non_null[k1] = {k2: v2 for k2, v2 in v1.items() if v2}

    if st.session_state.get("submit_ic"):
        st.session_state["inclusion_criteria"] = params_non_null
        st.success("Inclusion Criteria saved for the meta-review", icon="🔥")

    with st.sidebar:
        if st.session_state.get("hypotheses"):
            st.write("You have chosen the following hypotheses:")
            for hypothesis in st.session_state["hypotheses"]:
                st.write(hypothesis)
        if st.session_state.get("inclusion_criteria"):
            st.write("You have chosen the following inclusion criteria:")
            st.write(st.session_state["inclusion_criteria"])

if __name__ == '__main__':
    main()
