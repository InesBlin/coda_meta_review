# -*- coding: utf-8 -*-
"""
History of hypothesis already explored
"""
import os
import json
from collections import defaultdict
import streamlit as st

def main():
    """ Main """
    folder = "app/meta_review"
    mrs = sorted([x for x in os.listdir(folder) if x[0].isdigit()])
    h_type_h = defaultdict(list)
    for mr in mrs:
        if "type_h.txt" in os.listdir(os.path.join(folder, mr)):
            with open(os.path.join(folder, mr, "config.json"), "r", encoding="utf-8") as openfile:
                hypothesis = json.load(openfile)["hypothesis"]
            with open(os.path.join(folder, mr, "type_h.txt"), "r", encoding="utf-8") as openfile:
                type_h = openfile.read()
            if hypothesis not in h_type_h[type_h]:
                h_type_h[type_h].append(hypothesis)
    st.write("You have already explored the following hypotheses:")
    for k, v in h_type_h.items():
        st.write(f"### {k}")
        for h in v:
            st.markdown(f"- {h}")


if __name__ == '__main__':
    main()
