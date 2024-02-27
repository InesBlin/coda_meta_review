# -*- coding: utf-8 -*-
"""
All related to moderators (fetching, for meta-analysis)
"""
import re
import pandas as pd
from functools import reduce

def get_value_name(var, value_name):
    def extract_value(x):
        if not isinstance(x, str):
            x = str(x)
        split_values = x.split("|")
        return [value.split(" : ")[1] for value in split_values if var.lower() == value.split(" : ")[0].strip().lower()]
    
    values = [extract_value(x) for x in value_name]
    values = [",".join(v) if v else "NA" for v in values]
    return values

def define_moderators(mod, value_name1, value_name2):
    value1 = get_value_name(mod, value_name1)
    value2 = get_value_name(mod, value_name2)
    
    if any(pd.notna(value2)):
        print("1")
        values = pd.DataFrame({'value1': value1, 'value2': value2})
        values[mod] = "Treatment 1: " + values['value1'] + " vs. Treatment 2: " + values['value2']
        values = values[[mod]].map(lambda x: x.replace("NA", "none/NA") if isinstance(x, str) else x)
        print(values)
    elif all(pd.isna(value2)) and any([bool(re.match("^[A-Za-z]+$", str(v))) for v in value1]):
        print("2")
        values = pd.DataFrame({'value1': value1, 'value2': value2})
        values[mod] = "Treatment 1: " + values['value1']
        values = values[[mod]].map(lambda x: x.replace("NA", "none/NA") if isinstance(x, str) else x)
    elif all(pd.isna(value2)) and all([not bool(re.match("^[A-Za-z]+$", str(v))) for v in value1]):
        print("3")
        values = pd.DataFrame({'value1': value1, 'value2': value2})
        values[mod] = pd.to_numeric(values['value1'])
        values = values[[mod]]

    return values

def bind_moderators(mod, data):
    for moderator in mod:
        mod_def = define_moderators(moderator, data['treatmentValue1'], data['treatmentValue2'])
        data = pd.concat([data, mod_def], axis=1)

    return data

class ModeratorComponent:
    """
    From R Shiny app code, server.R, l573->l580
    # Function to create moderator variables for meta-analysis
    # `mod` takes a value from input$definemod
    # valueName1 and valueName2 take filteredObservationData$valueName1 and
    # filteredObservationData$valueName2
    # Creates three types of moderator variables:
    # 1) Both Treatment 1 and Treatment 2 have a value
    # 2) Only Treatment 1 has a non-numeric value
    # 3) Only Treatment 1 has a numeric value (continuous)

    From own understanding:
    - Possible moderators = intersection of
        (1) other specific independent variables (from the generic independent variables)
        (2) specific independent variables for which each study has a value
    """
    pass


if __name__ == '__main__':
    # Main
    from src.pipeline import Pipeline
    from kglab.helpers.data_load import read_csv
    OBS_DATA = read_csv("data/observationData.csv")
    # MOD = ["punishment incentive", "sequential punishment"]
    MOD = ["punishment incentive"]
    PIPELINE = Pipeline(siv1="punishment treatment", sivv1="1", siv2="punishment treatment", sivv2="-1")
    DF = PIPELINE.get_data_meta_analysis(data=OBS_DATA)
    DF = bind_moderators(mod=MOD, data=DF)
    DF.to_csv("test.csv")
