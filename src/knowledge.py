# -*- coding: utf-8 -*-
"""
Domain-expert knowledge
"""
import re

HYPOTHESES = {
    "regular": "{dependent} is significantly {comparative} when {iv_label} is {cat_t1_label} compared to when {iv_label} is {cat_t2_label}.",
    "var_mod": "When comparing studies where {iv_label} is {cat_t1_label} and studies where {iv_label} is {cat_t2_label}, effect sizes from studies involving {mod_t1_label} as {mod_label} are significantly {comparative} than effect sizes based on {mod_t2_label} as {mod_label}.",
    "study_mod": "When comparing studies where {iv_label} is {cat_t1_label} and studies where {iv_label} is {cat_t2_label}, cooperation is significantly {comparative} when {mod_label} is {mod_val_label} compared to when {mod_label} has another value.",
}

NE_HYPOTHESES = {
    "regular": "There is no significant difference in {dependent} when comparing studies where {iv_label} is {cat_t1_label} and studies where {iv_label} is {cat_t2_label}.",
    "var_mod": "When comparing studies where {iv_label} is {cat_t1_label} and studies where {iv_label} is {cat_t2_label}, there is no significant differences in effect sizes between studies involving {mod_t1_label} as {mod_label} and studies involving {mod_t2_label} as {mod_label}.",
    "study_mod": "When comparing studies where {iv_label} is {cat_t1_label} and studies where {iv_label} is {cat_t2_label}, there is no significant difference when {mod_label} is {mod_val_label} compared to when {mod_label} has another value.",
}

def generate_hypothesis(row, th):
    """ Generate human-readable hypothesis """
    if row.comparative in ['higher', 'lower']:
        template = HYPOTHESES[th]
    else:  # no predicted effect
        template = NE_HYPOTHESES[th]
    pattern = r'\{(.*?)\}'
    matches = re.findall(pattern, template)
    for col in matches:
        if col == "dependent":
            if row[col].endswith("s"):
                replace = row[col][:-1].split('/')[-1]
            else:
                replace = row[col].split('/')[-1]
            template = template.replace("{" + col + "}", replace)
        else:
            if row[col].split("/")[-1] != "na":
                to_replace = row[col].split("/")[-1]
            else:
                to_replace = row[col.replace("_label", "")]
            template = template.replace("{" + col + "}", to_replace)
    row["hypothesis"] = template.capitalize()
    return row
