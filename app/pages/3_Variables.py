# -*- coding: utf-8 -*-
""" Focus variables for the meta-review """
import os
import streamlit as st
import pandas as pd
from src.data_selection import DataSelector
from src.data_prep import DataPrep
from src.inclusion_criteria import InclusionCriteria
from src.settings import ROOT_PATH
from src.moderator import define_moderators

DATA = pd.read_csv(os.path.join(ROOT_PATH, "data/observationData.csv"), index_col=0)

if len(st.session_state.get("hypotheses")) == 0:
    st.warning("You haven't chosen a hypothesis yet. To do so, " + \
        "please refer to the page 'Select a hypothesis'.")
else:
    # Filtering data based on hypothesis + inclusion criteria 
    # (to ensure vars are consistant with data)
    h = st.session_state["hypotheses"][0]
    data_selector = DataSelector(siv1=h["siv1"], siv2=h["siv2"], sivv1=h["sivv1"], sivv2=h["sivv2"])
    data_prep = DataPrep(siv1=h["siv1"], sivv1=h["sivv1"], siv2=h["siv2"], sivv2=h["sivv2"])
    DATA = data_selector(data=DATA)
    DATA = data_prep(filtered_data=DATA)
    if st.session_state.inclusion_criteria:
        inclusion_criteria = InclusionCriteria(**st.session_state.inclusion_criteria)
        DATA = inclusion_criteria(data=DATA)

@st.cache_data
def get_variables(data):
    """ Get all variables """
    mr_variables = data.columns
    filter_out_startswith = ['observation', 'effectSize']
    mr_variables = [x for x in mr_variables if not any(x.startswith(s) \
        for s in filter_out_startswith)]
    filter_out = [
       'variance', 'study', 'studyName',
       'treatmentSubproperties', 'ivnames', 'valueNameSupport', 'studyName.y',
       'paper', 'DOI', 'paperDate', 'paperTitleDOI', 'authorNamesDOI',
       'country', 'studyAcademicDiscipline', 'yearOfDataCollection',
       'yearSource', 'treatmentValue1', 'treatmentValue2', 'paperName',
       'studyNameGeneral', 'paperYearDOI', 'substudy', 'paper_ID',
       'paperTitle', 'citation', 'paperYear', 'authorNames', 'lang']
    mr_variables = set(mr_variables).difference(set(filter_out))

    # sub_props = data[~data.treatmentSubproperties.isna()].treatmentSubproperties.values
    # sub_prop = set(y for x in sub_props for y in x.split(","))
    # mr_variables = set(mr_variables).union(sub_prop)
    return mr_variables

# @st.cache_data
# def add_var_info(data, variables):
#     """ Add info related to control variables """
#     f_vars = [x for x in variables if x not in data.columns]
#     for x in f_vars:
#         mod_def = define_moderators(x, data['treatmentValue1'], data['treatmentValue2'])
#         data = pd.concat([data, mod_def], axis=1)
#     return data



def main():
    """ Main """
    if "mr_variables" not in st.session_state:
        st.session_state["mr_variables"] = []
    if "submit_mr_variables" not in st.session_state:
        st.session_state["submit_mr_variables"] = False
    st.title("Variables")
    st.title("#")

    st.write("Choose your variables for the meta-review")

    variables = get_variables(data=DATA)

    with st.form("form_mr_variables"):
        mr_variables = st.multiselect(
            'Choose your variables for the meta-review',
            options=variables
        )
        if st.form_submit_button("Save variables"):
            st.session_state["submit_mr_variables"] = True
            st.session_state["mr_variables"] = mr_variables
            st.success("Variables added for the meta-review", icon="🔥")
            # data = add_var_info(data=DATA, variables=mr_variables)
            # st.write(data)


if __name__ == '__main__':
    main()
