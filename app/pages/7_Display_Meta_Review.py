# -*- coding: utf-8 -*-
""" Display the meta-review """
import os
import streamlit as st
import streamlit.components.v1 as components

def get_source_code(html_path: str) -> str:
    """ Return graph visualisation HTML """
    with open(html_path, 'r', encoding='utf-8') as html_file:
        source_code = html_file.read()
    return source_code

def main():
    """ Main """
    if "submit_h" not in st.session_state:
        st.session_state["submit_h"] = False
    mrs = sorted([x for x in os.listdir("app/meta_review") if x[0].isdigit()])
    if mrs and st.session_state.submit_h:
        path = os.path.join("app/meta_review", mrs[-1], "report.html")
        source_code = get_source_code(html_path=path)
        components.html(source_code, scrolling=False, height=10000)
    else:
        st.warning("It looks like your meta-review was not generated.")


if __name__ == '__main__':
    main()