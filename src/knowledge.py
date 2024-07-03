# -*- coding: utf-8 -*-
"""
Domain-expert knowledge
"""
import re

INVERSES = {"higher": "lower", "lower": "higher"}

HYPOTHESES = {
    "regular": "Cooperation is significantly {comparative} when {iv_label} is {cat_t1_label} compared to when {iv_label} is {cat_t2_label}.",
    "var_mod": "When comparing studies where {iv_label} is {cat_t1_label} and studies where {iv_label} is {cat_t2_label}, cooperation from studies involving {mod_t1_label} as {mod_label} is significantly {comparative} than cooperation from studies involving {mod_t2_label} as {mod_label}.",
    "study_mod": "When comparing studies where {iv_label} is {cat_t1_label} and studies where {iv_label} is {cat_t2_label}, cooperation is significantly {comparative} when {mod_label} is {mod_val_label} compared to when {mod_label} has another value.",
}

NE_HYPOTHESES = {
    "regular": "There is no significant difference in cooperation when comparing studies where {iv_label} is {cat_t1_label} and studies where {iv_label} is {cat_t2_label}.",
    "var_mod": "When comparing studies where {iv_label} is {cat_t1_label} and studies where {iv_label} is {cat_t2_label}, there is no significant differences in cooperation between studies involving {mod_t1_label} as {mod_label} and studies involving {mod_t2_label} as {mod_label}.",
    "study_mod": "When comparing studies where {iv_label} is {cat_t1_label} and studies where {iv_label} is {cat_t2_label}, there is no significant difference in cooperation when {mod_label} is {mod_val_label} compared to when {mod_label} has another value.",
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
        if col == "comparative":
            if "withdrawal" in row["dependent"].lower():
                to_replace = INVERSES[row.comparative]
            else:
                to_replace = row.comparative
            template = template.replace("{" + col + "}", to_replace)
        else:
            if row[col].split("/")[-1] != "na":
                to_replace = row[col].split("/")[-1]
            else:
                to_replace = row[col.replace("_label", "")]
            template = template.replace("{" + col + "}", to_replace)
    row["hypothesis"] = template.capitalize()
    return row
