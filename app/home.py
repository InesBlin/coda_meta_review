# -*- coding: utf-8 -*-
""" Streamlit app -> interactive part """
import os
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
from src.settings import ROOT_PATH


def main():
    """ Main """
    if "hypotheses" not in st.session_state:
        st.session_state["hypotheses"] = []
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.title("Generating Meta-Reviews with the CoDa Databank")
    st.markdown(
        """
        ## Introduction

        This interface enables you to generate your own meta-reviews based on your starting 
        hypothesis.

        The COoperation DAtabank (CoDa) dataset [1] "consists in 3,000 studies from the social
        and behavioural sciences published in 3 languages and annotated with more than 300 
        cooperation-related features, including characteristics of the sample participating in the 
        study (e.g. sample size, average age of sample, percentage of males, country of 
        participants), characteristics of the experimental paradigm (structure of the social 
        dilemma,incentives, repeated trial data), and quantitative results of the experiment (e.g. 
        mean levels of cooperation, variance in cooperation, and effect sizes)." [2]

        ## Hypotheses
        We have first worked with experts to design templated hypotheses that can be generated. 
        They are all based on the vocabulary (or ontology) on which the CODA dataset was built.
        The hypotheses and their templates are displayed below:
        """
    )
    pdf_viewer(
        input=os.path.join(ROOT_PATH, "app/figures/hypotheses-streamlit-app-home.drawio.pdf"))
    st.markdown(
        """
        From discussions with experts, we have also come up with the following 
        list of recommendations:
        1. It is more sensible to compare specific independent variables from the same generic
        independent variables;
        2. You can first start with a regular hypothesis, then explore finer-grained 
        moderator hypotheses.

        ## How to use this application?
        At the moment, only regular hypotheses are supported.
        You can navigate through the different tabs on the left to generate your meta-review.
        1. **Select a hypothesis.** Choose your variables for the hypothesis.
        2. **Inclusion Criteria.** Inclusion criteria for the studies.
        3. **Variables.** Control variables in the meta-review.
        4. **Analytic Strategy.** Statistical models.
        5. **Custom text.** Custom text to be added in the final meta-review.
        6. **Generate Meta Review** Generating the review.
        7. **Display Meta Review** Meta review visualisation.

        """
    )

    st.markdown(
        """
        ---
        [1] The Cooperation Databank: machine-readable science accelerates 
        research synthesis - Spadaro et al. - 2022

        [2] Discovering research hypotheses in social science using knowledge 
        graph embeddings - de Haan et al. - 2021
        """
    )

    with st.sidebar:
        st.write("You have chosen the following hypotheses:")
        for h in st.session_state["hypotheses"]:
            st.write(h)

if __name__ == '__main__':
    main()
