# -*- coding: utf-8 -*-
"""
All related to moderators (fetching, for meta-analysis)
"""
import re
from collections import defaultdict
import pandas as pd
from kglab.helpers.data_load import read_csv
from kglab.helpers.kg_query import run_query
from kglab.helpers.variables import HEADERS_CSV
from src.helpers import run_request, remove_url
from src.variables import NS_CDO, NS_CDP

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
        values = pd.DataFrame({'value1': value1, 'value2': value2})
        values[mod] = "Treatment 1: " + values['value1'] + " vs. Treatment 2: " + values['value2']
        values = values[[mod]].map(lambda x: x.replace("NA", "none/NA") if isinstance(x, str) else x)
    elif all(pd.isna(value2)) and any([bool(re.match("^[A-Za-z]+$", str(v))) for v in value1]):
        values = pd.DataFrame({'value1': value1, 'value2': value2})
        values[mod] = "Treatment 1: " + values['value1']
        values = values[[mod]].map(lambda x: x.replace("NA", "none/NA") if isinstance(x, str) else x)
    elif all(pd.isna(value2)) and all([not bool(re.match("^[A-Za-z]+$", str(v))) for v in value1]):
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
    def __init__(self,  api: str = "https://api.odissei.triply.cc/queries/coda-dev/",
                 sparql_endpoint: str = "http://localhost:7200/repositories/coda",
                 **cached):
        """
        cached: any cached data from SPARQL queries to avoid running them repeatedly
        - Useful if: running a lot of experiments, to avoid GraphDB heap memory errors
        """
        self.cached = cached if cached else {}
        self.api = api
        self.study_moderator_query = api + "moderators/1/run"
        self.study_moderators = self.get_study_moderators()

        self.sparql_endpoint = sparql_endpoint
        self.country_prop_query = """
        PREFIX cdo: <https://data.cooperationdatabank.org/vocab/class/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?p ?pLabel WHERE {
            ?s rdf:type cdo:Country ;
               ?p ?o .
            ?p rdfs:label ?pLabel.
            FILTER ( ?p != rdfs:label).
            FILTER ( ?p != rdf:type).
        }
        """
        # Simple country moderator: the value is accessible directly
        # Complex country moderator: value is accessible through an intermediate blank node
        self.country_moderators, self.country_mod_simple, self.country_mod_complex = \
            self.get_country_moderators()
        
        start, end = "{", "}"
        self.query_cmod_simple = f"""
        SELECT ?country (?value as ?<p_alt_name_replace>) WHERE {start}
            ?country <predicate_replace> ?value .
        {end}
        """
        self.query_cmod_complex = f"""
        PREFIX cdp: <https://data.cooperationdatabank.org/vocab/prop/>
        SELECT ?country (STR(?Year) as ?year) (?value as ?<p_alt_name_replace>) WHERE {start}
            ?country <predicate_replace> [
            cdp:year ?Year;
            cdp:value ?value
            ]
        {end}
        """
        

        start, end = "{", "}"
        self.variable_moderator_candidate_template = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX cdo: <https://data.cooperationdatabank.org/vocab/class/>
        PREFIX cdp: <https://data.cooperationdatabank.org/vocab/prop/>
        SELECT ?siv ?siv_label
        WHERE {start}
            ?siv rdfs:subPropertyOf ?giv ;
                rdfs:label ?siv_label .
            FILTER(?siv not in (cdp:<siv1>, cdp:<siv2>, cdp:<giv>)) .
            FILTER(?giv = cdp:<giv>) .  
        {end}
        ORDER BY ASC(?siv)
        """
    
    @staticmethod
    def get_count(data, id_treatment, candidates):
        res = set()
        for t_val in data[f"treatmentValue{id_treatment}"].values:
            t_val = t_val if isinstance(t_val, str) else str(t_val)
            curr_vars = t_val.split("|")
            for var in curr_vars:
                siv = var.split(" : ")[0].lower()
                if siv in candidates:
                    res.add(siv)
                if res == candidates:
                    return res
        return res
    
    def var_mods_one_treatment(self, data, siv1, siv2, giv, id_treatment):
        """ Variable treatment for one treatment"""

        if self.cached.get('variable_moderators'):
            candidates = read_csv(self.cached.get('variable_moderators'))
            candidates = candidates[(
                (candidates.giv == "https://data.cooperationdatabank.org/vocab/prop/" + giv) & \
                    (~candidates.siv.isin([f"https://data.cooperationdatabank.org/vocab/prop/{siv}" \
                        for siv in [siv1, siv2]])))]
            candidates = candidates[["siv", "siv_label"]]
        else:
            query = self.variable_moderator_candidate_template.replace("<giv>", giv) \
                .replace("<siv1>", siv1).replace("<siv2>", siv2)
            candidates = run_query(query=query, sparql_endpoint=self.sparql_endpoint, headers=HEADERS_CSV)
        candidates = candidates.drop_duplicates()
        candidates = {x.lower() for x in candidates.siv_label.unique()}
        t_val_count = self.get_count(data=data, id_treatment=id_treatment, candidates=candidates)
        candidates = candidates.intersection(t_val_count)
        return candidates

    
    def get_variable_moderators(self, data, info):
        """
        Variable moderators: server.R lx -> ly
        - data: pd.DataFrame with input data for meta-analysis
        - info: dict with the following keys: giv1, siv1, giv2, siv2
        """
        keys = ["giv1", "siv1", "giv2", "siv2"]
        if any (x not in info for x in keys):
            raise ValueError(f"All the followings should be in the keys of `info`: {keys}")
        res = set()
        for id_treatment in ["1", "2"]:
            res = res.union(self.var_mods_one_treatment(data=data, siv1= info["siv1"], siv2= info["siv2"],
                                                giv= info[f"giv{id_treatment}"], id_treatment=id_treatment))
        return res


    def get_study_moderators(self):
        """
        Study moderators: server.R l140->l145
        """
        if self.cached.get('study_moderators'):
            mod_from_kg = read_csv(self.cached.get('study_moderators'))
        else:
            mod_from_kg = run_request(
                self.study_moderator_query, headers={"Accept": "text/csv"})
        moderators = {"ageLow": "Age Low", "ageHigh": "Age High",
                      "choiceLow": "Lowest number of choices" ,
                      "choiceHigh": "Highest number of choices"}

        # Adding remaining columns from the DataFrame
        for index, row in mod_from_kg.iterrows():
            if index >= 2:
                moderators[row.moderator] = row.desc
        return moderators

    def get_country_moderators(self):
        """
        Country moderators: server.R lx -> ly
        """
        if self.cached.get('country_moderators'):
            moderators = read_csv(self.cached.get('country_moderators'))
        else:
            moderators = run_query(
                query=self.country_prop_query,
                sparql_endpoint=self.sparql_endpoint, 
                headers={"Accept": "text/csv"})
        start_filter = "https://data.cooperationdatabank.org/"
        moderators_simple = moderators[~moderators.p.str.startswith(start_filter)]
        moderators_complex = moderators[moderators.p.str.startswith(start_filter)]
        return moderators, moderators_simple, moderators_complex

    def add_country_moderator_unique(self, data, c_mod):
        """ Retrieve values to add in `data` for country moderator `c_mod`"""
        predicate = self.country_moderators[self.country_moderators.pLabel.str.lower() == c_mod].p.values[0]
        pred_alt_name = "_".join(c_mod.split())

        if c_mod in self.country_mod_simple.pLabel.str.lower().values:
            if self.cached.get('simple_country_moderators'):
                query_result = read_csv(self.cached.get('simple_country_moderators'))
                query_result = query_result[query_result.processedLabel == pred_alt_name]
                query_result = query_result[["country", "value"]]
                query_result.columns = ["country", pred_alt_name]
            else:
                query = self.query_cmod_simple.replace("predicate_replace", predicate) \
                    .replace("<p_alt_name_replace>", pred_alt_name)
                query_result = run_query(query=query, sparql_endpoint=self.sparql_endpoint,
                                         headers=HEADERS_CSV)
        else:  # c_mod in self.country_mod_complex.pLabel.str.lower().values
            if self.cached.get('complex_country_moderators'):
                query_result = read_csv(self.cached.get('complex_country_moderators'))
                query_result = query_result[query_result.processedLabel == pred_alt_name]
                query_result = query_result[["country", "year", "value"]]
                query_result.columns = ["country", "year", pred_alt_name]
            else:
                query = self.query_cmod_complex.replace("predicate_replace", predicate) \
                    .replace("<p_alt_name_replace>", pred_alt_name)
                query_result = run_query(query=query, sparql_endpoint=self.sparql_endpoint, headers=HEADERS_CSV)

        query_result['country'] = list(map(remove_url, query_result['country']))

        if c_mod in self.country_mod_simple.pLabel.str.lower().values:
            query_result[pred_alt_name] = list(map(remove_url, query_result[pred_alt_name]))
            query_result[pred_alt_name] = pd.to_numeric(query_result[pred_alt_name], errors='coerce')
            query_result.columns = ['country', c_mod]
        else:
            query_result['year'] = query_result['year'].astype(int)
            query_result.columns = ['country', 'year', c_mod]

        data["country"] = data['country'].astype(str)
        query_result["country"] = query_result['country'].astype(str)
        return data.merge(query_result, on='country')
    
    def add_country_mods(self, data, mods):
        """ Add country moderator """
        for mod in mods:
            data = self.add_country_moderator_unique(data=data, c_mod=mod)
        return data



if __name__ == '__main__':
    # Main
    # from src.pipeline import Pipeline
    # from kglab.helpers.data_load import read_csv
    # OBS_DATA = read_csv("data/observationData.csv")
    # # MOD = ["punishment incentive", "sequential punishment"]
    # MOD = ["punishment incentive"]
    # PIPELINE = Pipeline(siv1="punishment treatment", sivv1="1", siv2="punishment treatment", sivv2="-1")
    # DF = PIPELINE.get_data_meta_analysis(data=OBS_DATA)
    # DF = bind_moderators(mod=MOD, data=DF)
    
    from rdflib import Namespace
    from kglab.helpers.data_load import read_csv
    CACHED = {
        "study_moderators": "./data/moderators/study_moderators.csv",
        "country_moderators": "./data/moderators/country_moderators.csv",
        "simple_country_moderators": "./data/moderators/simple_country_moderators.csv",
        "complex_country_moderators": "./data/moderators/complex_country_moderators.csv",
        "variable_moderators": "./data/moderators/variable_moderators.csv"
    }
    MODERATOR_C = ModeratorComponent(**CACHED)
    NS_CDO = Namespace("https://data.cooperationdatabank.org/vocab/class/")
    DATA = read_csv("./data/observationData.csv")
    CM = MODERATOR_C.get_variable_moderators(
        data=DATA, info={"siv1": "punishmentTreatment", "siv2": "rewardIncentive",
                         "giv1": "PunishmentVariable", "giv2": "RewardVariable"})
    print(MODERATOR_C.country_moderators)
    print(CM, len(CM))
    print(MODERATOR_C.cached)
    

    DATA = MODERATOR_C.add_country_mods(data=DATA, mods=["airports", "eastern church exposure"])
    # print(DATA)