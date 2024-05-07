# -*- coding: utf-8 -*-
""" Streamlit app -> interactive part """
import streamlit as st

def main():
    """ Main """
    if "hypotheses" not in st.session_state:
        st.session_state["hypotheses"] = []
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.title("Generating Meta-Reviews with the CoDa Databank")


    st.markdown("## Introduction ")
    st.markdown("### What is the CoDa Databank?")
    st.write("[CONTENT]")
    st.markdown("### What is a meta-review?")
    st.write("[CONTENT]")
    st.markdown("### How to use this application?")
    st.write("[CONTENT]")

    st.markdown("## Hypotheses explored")
    st.write("#")
    st.write("In this interface, you will be able to generate meta-reviews from three types of hypothesis:")
    st.markdown("* REGULAR: [CONTENT]")
    st.markdown("* NUMERICAL: [CONTENT]")
    st.markdown("* CATEGORICAL: [CONTENT]")
    st.write("From discussions with experts, we have come up with the following list of recommendations:")
    st.markdown("1.  No cross-bubble is more sensible")
    st.markdown("2. First regular hypothesis, then fine-grained with moderators")

    with st.sidebar:
        st.write("You have chosen the following hypotheses:")
        for h in st.session_state["hypotheses"]:
            st.write(h)

if __name__ == '__main__':
    main()
