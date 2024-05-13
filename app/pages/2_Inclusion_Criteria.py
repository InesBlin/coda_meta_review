# -*- coding: utf-8 -*-
""" Inclusion Criteria """

import streamlit as st
import pandas as pd
from src.inclusion_criteria import InclusionCriteria

IC = InclusionCriteria()
IC_TYPES = sorted(IC.mappings.keys())
DATA = pd.read_csv("./data/observationData.csv")
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
    correct_vals = sorted([float(val) for val in vals \
        if isinstance(val, (int, float)) or \
            (isinstance(val, str) and val.isdigit())])
    return correct_vals[0], correct_vals[-1]


def main():
    """ Main for inclusion criteria """
    if 'inclusion_criteria' not in st.session_state:
        st.session_state["inclusion_criteria"] = None

    st.title("Inclusion Criteria")
    st.write("#")

    if not st.session_state["hypotheses"]:
        st.warning("You haven't selected a hypothesis/several hypotheses yet. Please enter one in the `Select a hypothesis` section.", icon="🚨")

    st.write("You can now choose additional inclusion criteria.")
    st.write("If you don't want to add inclusion criteria, please click the button `Confirm my inclusion criteria` directly.")
    type_ic = st.multiselect(
        "Which inclusion criteria would you like to add?",
        IC_TYPES
    )
    params = {k: {} for k in type_ic}
    for tic in type_ic:
        [_, _, simple_ic, range_ic] = IC.mappings[tic]
        st.write(simple_ic)
        st.write(range_ic)
        with st.form(f"ic_{tic}"):
            for simple in simple_ic:
                params[tic][simple] = st.multiselect(
                    f'Choose your inclusion criteria for `{simple}`',
                    options=get_options_simple_ic(data=DATA, val=simple)
                )
            for range_ in range_ic:
                min_range, max_range = get_range_ic(data=DATA, val=range_)
                params[tic][range_] = st.slider(
                    f"Select your range of values for {range_}",
                    min_range, max_range, (min_range, max_range)
                )
            if st.form_submit_button(f"Save {tic} inclusion criteria"):
                st.session_state[f"submit_ic_{tic}"] = True
    
    params_non_null = {}
    for k1, v1 in params.items():
        params_non_null[k1] = {k2: v2 for k2, v2 in v1.items() if v2}
    
    if st.button("Confirm my inclusion criteria"):
        st.session_state["inclusion_criteria"] = params_non_null


if __name__ == '__main__':
    main()