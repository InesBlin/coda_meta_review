# -*- coding: utf-8 -*-
"""
server.R -> line 1031-> 1164

Takes pre-selected data, and does some preprocessing for the meta-analysis
"""
import click
from kglab.helpers.data_load import read_csv
from src.helpers import select_observations

class DataPrep:
    """ Post-processing observation to be used for meta-analysis 
    
    In ui.R on the Coda R shiny app
    Treatment 1
    * Generic Independent Variable -> genIVselection1a
    * Specific Independent Variable -> treatmentSubpropSelection1a
    * Specific Independent Variable values -> valueOptionsSelection1a """
    def __init__(self, siv1, sivv1, siv2, sivv2):
        self.siv1 = siv1
        self.sivv1 = sivv1
        self.siv2 = siv2
        self.sivv2 = sivv2

        # Below: values (siv, var, level1, level2, level3) to apply the self.match_level function
        self.match_level_values = [
            ('age', "age cohort", "young", "middle", "old"),
            ("academic grade", "academic grade level", "junior", "middle", "senior"),
            ("expectations", "expectations level", "low", "medium", "high"),
            ('identification', "identification level", "low", "medium", "high"),
            ('entitativity', "entitativity level", "low", "medium", "high"),
            ('hormone', "hormone level", "low", "medium", "high"),
            ('social capital', "social capital level", "low", "medium", "high"),
            ('state trust', "state trust level", "low", "medium", "high"),
            ("participant's own behavior (correlation)", "participant's behavior level",
             "low", "medium", "high"),
            ("partner's behavior correlation", "partner's behavior level", "low", "medium", "high"),
            ('game comprehension', "game comprehension level", "low", "medium", "high"),
            ("participant's behavior (correlation)", "participant's behavior level",
             "low", "medium", "high"),
        ]

    @staticmethod
    def evaluate_list(str_input, criteria):
        """ Check whether a criteria is in elements of a list """
        return criteria in str_input.lower() if isinstance(str_input, str) else False

    def check_level(self, val, var, level):
        """ Check whether a variable has a certain level """
        criteria = f"{var} : {level}"
        return self.evaluate_list(val, criteria)

    def match_level(self, var, level1, level2, level3, filtered_observation_data):
        """ Check whether the levels of treatments are
        ordered in line with the individual difference variable """
        def case_when_conditions(row):
            if not self.evaluate_list(row['treatmentValue1'], var):
                return None
            if self.check_level(row['treatmentValue1'], var, level3):
                return True
            if self.check_level(row['treatmentValue1'], var, level2) and \
                self.check_level(row['treatmentValue2'], var, level1):
                return True
            if self.check_level(row['treatmentValue1'], var, level2) and \
                self.check_level(row['treatmentValue2'], var, level3):
                return False
            if self.check_level(row['treatmentValue1'], var, level1):
                return False
            return None

        filtered_observation_data['match'] = filtered_observation_data \
            .apply(case_when_conditions, axis=1)
        return filtered_observation_data

    def match_level_svo_unclassified(self, data):
        """ server.R -> l.1062: how to deal with unclassified? """
        def case_when_conditions(row):
            if not self.evaluate_list(row['treatmentValue1'], "svo type"):
                return None
            if self.check_level(row['treatmentValue1'], "svo type", "altruist"):
                return True
            if self.check_level(row['treatmentValue1'], "svo type", "prosocial") & \
                ((self.check_level(row['treatmentValue2'], "svo type", "individualist")) or \
                    (self.check_level(row['treatmentValue2'], "svo type", "proself")) or \
                        (self.check_level(row['treatmentValue2'], "svo type", "competitor"))):
                return True
            if self.check_level(row['treatmentValue1'], "svo type", "individualist") & \
                ((self.check_level(row['treatmentValue2'], "svo type", "proself")) or \
                    (self.check_level(row['treatmentValue2'], "svo type", "competitor"))):
                return True
            if self.check_level(row['treatmentValue1'], "svo type", "proself") & \
                self.check_level(row['treatmentValue2'], "svo type", "competitor"):
                return True
            if self.check_level(row['treatmentValue1'], "svo type", "proself") & \
                ((self.check_level(row['treatmentValue2'], "svo type", "individualist")) or \
                    (self.check_level(row['treatmentValue2'], "svo type", "prosocial")) or \
                        (self.check_level(row['treatmentValue2'], "svo type", "altruist"))):
                return False
            if self.check_level(row['treatmentValue1'], "svo type", "individualist") & \
                ((self.check_level(row['treatmentValue2'], "svo type", "prosocial")) or \
                    (self.check_level(row['treatmentValue2'], "svo type", "altruist"))):
                return False
            if self.check_level(row['treatmentValue1'], "svo type", "prosocial") & \
                self.check_level(row['treatmentValue2'], "svo type", "altruist"):
                return False
            if self.check_level(row['treatmentValue2'], "svo type", "altruist"):
                return False
            if self.check_level(row['treatmentValue1'], "svo type", "competitor"):
                return False
            return None

        data['match'] = data.apply(case_when_conditions, axis=1)
        return data

    def match_level_decision_time(self, data):
        """ server.R -> l.113è -> decision time """
        def case_when_conditions(row):
            if not self.evaluate_list(row['treatmentValue1'], "decision time"):
                return None
            if self.check_level(row['treatmentValue1'], "decision time", "fast"):
                return True
            if self.check_level(row['treatmentValue1'], "decision time", "slow"):
                return False
            return None

        data['match'] = data.apply(case_when_conditions, axis=1)
        return data


    @staticmethod
    def update_order_match_single(curr_match, order_match):
        """ Updating values in order_match based on the ones in curr_match 
        - curr_match and order_match have the same size
        - for index i:
            - if curr_match[i] is None -> order_match[i] unchanged
            - else -> order_match[i] becomes curr_match[i] """
        res = []
        for curr_val, order_val in zip(curr_match, order_match):
            if curr_val is None:
                res.append(order_val)
            else:
                res.append(curr_val)
        return res

    def update_order_match_all(self, df_input, order_match):
        """
        - df_input: df with still some data to post process
        
        """
        if (self.siv1 == "individual difference") & (self.sivv1 != "social value orientation"):
            df_input = self.match_level(
                var="individual difference level", level1="low", level2="medium", level3="high",
                filtered_observation_data=df_input)
            order_match = self.update_order_match_single(
                curr_match=df_input.match, order_match=order_match)

        if self.sivv1 == "social_value_orientation":
            df_input = self.match_level_svo_unclassified(data=df_input)
            order_match = self.update_order_match_single(
                curr_match=df_input.match, order_match=order_match)

        for (siv, var, level1, level2, level3) in self.match_level_values:
            if self.sivv1 == siv:
                df_input = self.match_level(var=var, level1=level1, level2=level2, level3=level3,
                                            filtered_observation_data=df_input)
                order_match = self.update_order_match_single(
                curr_match=df_input.match, order_match=order_match)

        if self.sivv1 == "decision time (correlation)":
            df_input = self.match_level_decision_time(data=df_input)
            order_match = self.update_order_match_single(
                curr_match=df_input.match, order_match=order_match)

        return order_match

    @staticmethod
    def reverse_effect_size_estimate(data, order_match):
        """ Reverse effect size estimate """
        data['match'] = order_match
        data['effectSize'] = data.apply(lambda row: row['effectSize'] if row['match'] \
            else -1 * row['effectSize'], axis=1)
        data['LL'] = data['effectSizeLowerLimit'].copy()
        data['UL'] = data['effectSizeUpperLimit'].copy()
        data['effectSizeLowerLimit'] = data.apply(lambda row: row["LL"] if row['match'] \
            else -1 * row["UL"], axis=1)
        data['effectSizeUpperLimit'] = data.apply(lambda row: row["UL"] if row['match'] \
            else -1 * row["LL"], axis=1)
        return data.drop(columns=["match", "LL", "UL"])

    @staticmethod
    def reverse_value_name(data, order_match):
        """ Reverse treatmentValue names, when applicable """
        data['match'] = order_match
        data['vN1'] = data.apply(lambda row: row['treatmentValue1'] if row['match'] \
            else row['treatmentValue2'], axis=1)
        data['vN2'] = data.apply(lambda row: row['treatmentValue2'] if row['match'] \
            else row['treatmentValue1'], axis=1)
        data.drop(columns=['treatmentValue1', 'treatmentValue2', 'match'], inplace=True)
        data.rename(columns={'vN1': 'treatmentValue1', 'vN2': 'treatmentValue2'}, inplace=True)
        return data


    def __call__(self, filtered_data):
        """ Main: 
        - data: pre-selected observations, with some post-processing to-do 
        - output: input to the meta-analysis """
        order_match = select_observations(
            data=filtered_data, siv=self.siv1, sivv=self.sivv1, treatment_number=1)

        # order_match = data[np.array(observations, dtype=bool)]
        order_match = self.update_order_match_all(df_input=filtered_data, order_match=order_match)
        filtered_data = self.reverse_effect_size_estimate(
            data=filtered_data, order_match=order_match)
        filtered_data = self.reverse_value_name(data=filtered_data, order_match=order_match)

        # filtered_data = filtered_data[np.array(order_match, dtype=bool)]
        return filtered_data


@click.command()
@click.option("--input_data_path", help="Link to selected data, .csv format")
@click.option("--save_path", help="path to save selected data")
def main(input_data_path, save_path):
    """ Main script to store obs data """
    data_prep = DataPrep(siv1="punishment treatment", sivv1="1",
                         siv2="punishment treatment", sivv2="-1")
    data = read_csv(input_data_path)
    data = data_prep(data)
    data.to_csv(save_path)

if __name__ == '__main__':
    main()
