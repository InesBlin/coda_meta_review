# -*- coding: utf-8 -*-
""" Streamlit app -> interactive part """
import streamlit as st
# from pages import home

# PAGES = {
#     "Home": home,
# }

# st.sidebar.title('Navigation')
# selection = st.sidebar.radio("Go to", list(PAGES.keys()))
# st.session_state["data_in_cache"] = True

# page = PAGES[selection]
# page.app()

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.title("Exploring the CoDa Databank")
st.write("#")
st.write("Explaning the purpose of this application")
