# -*- coding: utf-8 -*-
""" Gathering observation data """
import re
from copy import deepcopy
import click
import pandas as pd
from src.helpers.helpers import run_request


class KGDataCall:
    """ Class to retrieve observation data, to be used as basis for meta-analysis """
    def __init__(self, api: str = "https://api.odissei.triply.cc/queries/coda-dev/"):
        self.api = api

        self.page_observation = 3
        self.observation_query = api + "dashboard/1/run?page={}&pageSize=10000"

        self.page_support = 3
        self.page_query = api + "dashboard-support/1/run?page={}&pageSize=10000"

        self.study_query = f"{api}study-characteristics/1/run"

    def get_treatment_value(self, data):
        """ Format treatment values """
        # Only keeping observation and valueNameSupport, splitting by treatment and valuepairs
        curr_data = deepcopy(data[['observation', 'valueNameSupport']])
        curr_data['valueNameSupport'] = curr_data['valueNameSupport'].str.split('\\|')
        curr_data = curr_data.explode('valueNameSupport')
        temp_df = curr_data['valueNameSupport'].str.split(' ~ ', expand=True)
        temp_df.columns = ['treat', 'valuepairs']
        curr_data = curr_data.drop('valueNameSupport', axis=1).join(temp_df).drop_duplicates()

        # Separating treatment 1 values and treatment 2 values
        curr_data = curr_data.groupby(['observation', 'treat']) \
            .agg({'valuepairs': '|'.join}).reset_index()
        curr_data['treatprops'] = curr_data['treat'] + "~" + curr_data['valuepairs']
        curr_data = curr_data[['observation', 'treat', 'treatprops']].drop_duplicates()
        curr_data = curr_data.groupby(['observation']) \
            .agg({'treatprops': '<sep>'.join}).reset_index()

        temp_df = curr_data['treatprops'].str.split('<sep>', expand=True)
        temp_df.columns = ['treatmentValue1', 'treatmentValue2']
        curr_data = curr_data.drop('treatprops', axis=1).join(temp_df)


        # Post processing on treatment value 1 and 2
        curr_data['treatmentValue1'] = \
            curr_data['treatmentValue1'].str.replace('.*~', '', regex=True)
        curr_data['treatmentValue2'] = \
            curr_data['treatmentValue2'].str.replace('.*~', '', regex=True)

        data = data.merge(curr_data, how="inner", on="observation")

        # # Replace NA in a range of columns with empty strings
        cols_to_replace_na = data.loc[:, 'treatmentValue1':'treatmentValue2'].columns
        data[cols_to_replace_na] = data[cols_to_replace_na].fillna('')
        return data

    def add_additional_col(self, data, paper_id_path):
        """ Additional info related to paper info """
        # Create new columns based on substrings of existing columns
        data['paperName'] = data['observationName'].str.slice(0, 8)
        data['studyNameGeneral'] = data['studyName'].str.slice(6, 16)
        data['paperYear'] = data['paperDate'].str.slice(0, 4)

        # Create a new column based on the length of 'studyName'
        data['substudy'] = data['studyName'].apply(
            lambda x: 0 if len(x) == 16 else 1 if len(x) == 17 else None)

        # Replace specific string values across the entire DataFrame
        data = data.replace("False", "FALSE")
        data = data.replace("True", "TRUE")
        data = data.replace("False,True", "FALSE,TRUE")


        citations = pd.read_csv(paper_id_path)
        curr_data = citations[['paper_ID', 'Title', 'Authors & year']].copy()
        curr_data.rename(columns={'Title': 'paperTitle', 'Authors & year': 'citation'},
                         inplace=True)

        # # Extracting paper year and author names using regular expressions
        curr_data['paperYear'] = curr_data['citation'].apply(lambda x: re.findall(r'\d+', x))
        curr_data['authorNames'] = curr_data['citation'].apply(
            lambda x: re.sub(r', &', ',', re.sub(r' &', ',', re.sub(r' \(\w+\)', '', x))) \
                .replace(', ', ','))

        # # Unnesting the paperYear list into separate rows
        curr_data = curr_data.explode('paperYear')

        # # Merging the two DataFrames
        data = data.merge(curr_data, left_on='paperName', right_on='paper_ID',
                          how='left', suffixes=('DOI', ''))
        return data

    def get_observation_data(self, paper_id_path: str):
        """ Gather observation data, similar as in Coda R Shiny App """
        # Get data from observation query
        curr_df = []
        for i in range(1, self.page_observation + 1):
            curr_df.append(run_request(self.observation_query.format(i),
                                       headers={"Accept": "text/csv"}))
        obs_data = pd.concat(curr_df)
        obs_data['observationName'] = obs_data['observationName'].apply(lambda x: x[16:])

        # Adding information from support + info query
        curr_df = []
        for i in range(1, self.page_support + 1):
            curr_df.append(run_request(self.page_query.format(i),
                                       headers={"Accept": "text/csv"}))
        support_data = pd.concat(curr_df)
        support_data['observationName'] = support_data['observationName'].apply(lambda x: x[16:])
        obs_data = obs_data.merge(support_data, on="observationName", how="left")

        study_info = run_request(uri=self.study_query, headers={"Accept": "application/json"})
        obs_data = obs_data.merge(study_info, on='study', how='inner', suffixes=('', '.y'))

        # Retrieve value name support
        obs_data = self.get_treatment_value(data=obs_data)
        # Add info related to paper ID
        obs_data = self.add_additional_col(data=obs_data, paper_id_path=paper_id_path)

        return obs_data


@click.command()
@click.option("--paper_id", help="Link to paper_id .csv")
@click.option("--save_path", help="path to save observation data")
def main(paper_id, save_path):
    """ Main script to store obs data """
    kg_data_call = KGDataCall()
    obs_data = kg_data_call.get_observation_data(paper_id_path=paper_id)
    obs_data.to_csv(save_path)


if __name__ == '__main__':
    main()
