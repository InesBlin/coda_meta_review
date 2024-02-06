# -*- coding: utf-8 -*-
"""
generic helpers
"""
def evaluate_list(input_, criteria):
    """ returns boolean vector of whether a criteria is
    in each element of datalist """
    def helper(x):
        return criteria in x.lower() if isinstance(x, str) else False
    if isinstance(input_, list):
        return [helper(x) for x in input_]
    # Else a string
    return helper(input_)

def select_observations(data, siv, sivv, treatment_number):
    """ specific applications of evaluate_list """
    treatment_selection = f"{siv} : {sivv}"
    treatment_value_key = f"treatmentValue{treatment_number}"
    return evaluate_list(list(data[treatment_value_key]), treatment_selection)
