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


Other filtering that can be done in the app
server.R file from line 687 to 1017
- authors -> str regex on citation
- language -> str regex on observation name
- sample size -> range filter on overallN
- publication status -> value filter on publicationStatus
- deception, study symmetric, study known end game, study dilemna type, study continuous PGG, 
study experimental setting, study one shot, study one shot repeated, study matching protocol, 
study show up fee, study game incentive, discussion, participant decision, study real partner, 
study acquaintance, sanction, study number of choices, study choice low, study choice high, 
study kindex, study mpcr, study group size, replenishment rate, study pdgthreshold, etc

"""
import click
import numpy as np
from src.helpers import select_observations
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
    def __init__(self, siv1, sivv1, siv2, sivv2):
        self.siv1 = siv1
        self.sivv1 = sivv1
        self.siv2 = siv2
        self.sivv2 = sivv2

    def __call__(self, data):
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
@click.option("--obs_data_path", help="Link to all observation data, .csv format")
@click.option("--save_path", help="path to save selected data")
def main(obs_data_path, save_path):
    """ Main script to store obs data """
    data_selector = DataSelector(siv1="punishment treatment", sivv1="1",
                                 siv2="punishment treatment", sivv2="-1")
    data = read_csv(obs_data_path)
    data = data_selector(data)
    data.to_csv(save_path)


if __name__ == '__main__':
    main()
