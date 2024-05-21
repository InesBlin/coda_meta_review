# -*- coding: utf-8 -*-
"""
Filtering data based on treatment selection

Input = set of all observations
Output = subset of these observations

Starting hypothesis: comparing two treatments

As in the website
Treatment 1
- Generic Independent Variable
- Specific Independent Variable
- Specific Independent Variable value

Treatment 2
- Generic Independent Variable
- Specific Independent Variable
- Specific Independent Variable value
"""
import click
import numpy as np
from src.helpers.helpers import select_observations
from kglab.helpers.data_load import read_csv

def filter_sivv(row, col_name, val_1, val_2):
    """ Filter row, based on value of col_name 
    must compare val_1 vs. val_2 or val_2 vs. val_1
    """
    x = row[col_name]
    if not isinstance(x, str):
        return False
    split = [y.split("|") for y in x.split(" vs. ")]

    if (val_1 in split[0] and val_2 in split[1]) or (val_1 in split[1] and val_2 in split[0]):
        return True
    return False


class DataSelector:
    """  Selecting/Filtering/Preparing data for meta-analysis
    For now based on the all_data_coda_app.csv file 
    Later: directly from SPARQL """
    def __init__(self, siv1: str, sivv1: str, siv2: str, sivv2: str):
        """
        - `siv1`, `sivv1`, `siv2` and `sivv2`: literals representing the two treatments 
        - `inclusion_criteria`: if not null, a dict with the following possible keys
            for the inclusion criteria -> `sample`, `study`, `quantitative`, and/or `metadata`
        """
        self.siv1 = siv1
        self.sivv1 = sivv1
        self.siv2 = siv2
        self.sivv2 = sivv2

    def __call__(self, data):

        """ Filter based on T1/T2 values """
        so1_1 = select_observations(data, siv=self.siv1, sivv=self.sivv1, treatment_number=1)
        so1_2 = select_observations(data, siv=self.siv2, sivv=self.sivv2, treatment_number=2)
        so1 = np.array(so1_1) & np.array(so1_2)

        so2_1 = select_observations(data, siv=self.siv1, sivv=self.sivv1, treatment_number=2)
        so2_2 = select_observations(data, siv=self.siv2, sivv=self.sivv2, treatment_number=1)
        so2 = np.bitwise_and(so2_1, so2_2)

        filter_ = np.bitwise_or(so1, so2)
        data = data[np.array(filter_, dtype=bool)]
        return data


@click.command()
# "Link to all observation data, .csv format"
@click.argument("obs_data_path")
# path to save selected data
@click.argument("save_path")
def main(obs_data_path, save_path):
    """ Main script to store obs data """
    data_selector = DataSelector(siv1="punishment treatment", sivv1="1",
                                 siv2="punishment treatment", sivv2="-1")
    data = read_csv(obs_data_path)
    data = data_selector(data)
    data.to_csv(save_path)


if __name__ == '__main__':
    main()
