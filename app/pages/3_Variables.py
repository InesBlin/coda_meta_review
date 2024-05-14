# -*- coding: utf-8 -*-
""" Focus variables for the meta-review """
import streamlit as st
import pandas as pd

DATA = pd.read_csv("./data/observationData.csv", index_col=0)

@st.cache_data
def get_variables(data):
    """ Get all variables """
    mr_variables = data.columns
    filter_out_startswith = ['observation', 'effectSize']
    mr_variables = [x for x in mr_variables if not any(x.startswith(s) for s in filter_out_startswith)]
    filter_out = [
       'variance', 'study', 'studyName',
       'treatmentSubproperties', 'ivnames', 'valueNameSupport', 'studyName.y',
       'paper', 'DOI', 'paperDate', 'paperTitleDOI', 'authorNamesDOI',
       'country', 'studyAcademicDiscipline', 'yearOfDataCollection',
       'yearSource', 'treatmentValue1', 'treatmentValue2', 'paperName',
       'studyNameGeneral', 'paperYearDOI', 'substudy', 'paper_ID',
       'paperTitle', 'citation', 'paperYear', 'authorNames', 'lang']
    mr_variables = set(mr_variables).difference(set(filter_out))

    sub_props = data[~data.treatmentSubproperties.isna()].treatmentSubproperties.values
    sub_prop = set(y for x in sub_props for y in x.split(","))
    mr_variables = set(mr_variables).union(sub_prop)
    return mr_variables

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


if __name__ == '__main__':
    main()