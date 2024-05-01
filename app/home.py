# -*- coding: utf-8 -*-
""" Streamlit app -> interactive part """
import streamlit as st
# import signal

# def signal_handler(signal, frame):
#     print("Signal handler called with signal", signal)

# signal.signal(signal.SIGINT, signal_handler)

def main():
    """ Main """
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.title("Exploring the CoDa Databank")
    st.write("#")
    st.write("Explaning the purpose of this application")

if __name__ == '__main__':
    main()
