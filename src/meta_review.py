# -*- coding: utf-8 -*-
"""
Meta-review generation in Python

Query to get labels+description

```SPARQL
PREFIX dct: <http://purl.org/dc/terms/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT * WHERE {
    ?node rdfs:label ?label ;
          dct:description ?description .
}
```
"""
import os
import re
import math
from typing import Union, Dict
import click
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import plotly.express as px
from src.pipeline import Pipeline
from src.helpers.helpers import load_json_file, load_yaml_file

NE_HYPOTHESES = {
    "regular": """
        There is no significant difference in cooperation when comparing studies 
        where {siv1} is {sivv1} and studies where {siv2} is {sivv2}.""",
    "var_mod": """
        When comparing studies where {siv1} is {sivv1} and studies where {siv2} 
        is {sivv2}, there is no significant differences in cooperation between 
        studies involving {mod1} as {mod} and studies involving {mod2} as {mod}.""",
    "study_mod": """
        When comparing studies where {siv1} is {sivv1} and studies where 
        {siv2} is {sivv2}, there is no significant difference in cooperation when 
        {mod} is {mod1} compared to when {mod} has another value.""",
}

def generate_hypothesis(h, th):
    """ Generate human-readable hypothesis (no significant difference) """
    template = NE_HYPOTHESES[th]
    pattern = r'\{(.*?)\}'
    matches = re.findall(pattern, template)
    for col in matches:
        template = template.replace("{" + col + "}", h[col])
    return template.capitalize().strip()


def custom_enumerate(iterable, start=1):
    """ Custom enumerate for HTML """
    return enumerate(iterable, start)


def categorize_es(x):
    """ categorize effect size """
    if abs(x) >= 0.8:
        return "large"
    if abs(x) >= 0.5:
        return "medium"
    return "small"


def replace_cite_id(input_text, dico):
    """ In text, replace ~\\cite{} with [nb] """
    pattern = r'<cite (.*?)>'
    def replace_with_dict(match):
        key = match.group(1)
        return f"[{dico.get(key, '')}]"
    return re.sub(pattern, replace_with_dict, input_text)


def generate_sent_ic(inclusion_criteria):
    """ Generate sentence for inclusion criteria """
    if inclusion_criteria:
        return "For studies to be included in the meta-analysis, " + \
            "the following criteria had to be fulfiled:"
    return ""


class MetaReview:
    """ Main class for generating the meta-review
    Input = one hypothesis, can be of three types:
    1- Regular
    `Cooperation is significantly {higher} when {siv1} is {sivv1} 
    compared to when {siv2} is {sivv2}.`
    2- Numerical moderator
    `When comparing studies where {siv1} is {sivv1} and studies where {siv2} is {sivv2},
    cooperation is significantly {higher} with {high} values of {mod-value}.`",`
    3- Categorical moderator
    `When comparing studies where {siv1} is {sivv1} and studies where {siv2} is {sivv2},
    effect size from studies involving {mod1} as {mod} are significantly {higher}
    than effect sizes based on {mod2} as {mod}.`
    (cf. below for templated hypotheses)
    
    Output = html file with the meta-review
    Ending of the output = whether the hypothesis is validated or not 
    
    Hypothesis is in dictionary format
    1-, 2-, and 3- all contain `siv1`, `sivv1`, `siv2` and `sivv2` as keys 
    1- contains `'comparative'` as key
    2- contains `'comparative'`, `mod_qualifier`, `mod`,  `type_mod` as keys
    3- contains `mod1`, `mod`, `comparative`, `mod2`, `type_mod` as keys

    [NB]: for now focus on 1-
    """

    def __init__(self, config: Union[Dict, str]):
        """ Init main templates/variables """

        if isinstance(config, str):
            config = load_json_file(filename=config)

        if isinstance(config["data"], str):
            config["data"] = pd.read_csv(config["data"], index_col=0)


        self.data = config["data"]
        self.h = config["hypothesis"]
        self.label_des = pd.read_csv(config["label_des"])

        if "config" in config:
            self.config = load_yaml_file(config["config"])
        else:
            keys_config = [
                "title", "authors", "introduction", "inclusion_criteria_1",
                "inclusion_criteria", "es_measure",
                "control_variables", "method_mv"]
            self.config = {k: config.get(k) for k in keys_config}

        self.references = load_json_file(config["references"])
        self.id_ref_to_nb = {x["id"]: i+1 for i, x in enumerate(self.references)}
        self.config = {k: replace_cite_id(v, self.id_ref_to_nb) \
            if isinstance(v, str) else v for k, v in self.config.items()}
        self.type_h = self.get_type_hypothesis()

        self.h_template = {
            "regular": "Cooperation is significantly {} when {} is {} compared to when {} is {}.",
            "numerical_moderator": "When comparing studies where {} is {} and studies " + \
                "where {} is {}, cooperation is significantly {} with {} values of {}.",
            "categorical_moderator": "When comparing studies where {} is {} and studies "+ \
                "where {} is {}, effect size from studies involving {} as {} are " + \
                    "significantly {} than effect sizes based on {} as {}."
        }
        self.h_template_not_stat_sig = {k: v.replace("significantly", "") \
            for k, v in self.h_template.items()}
        self.text_h = self.get_text_hypothesis(templates=self.h_template, hyp=self.h)
        self.inclusion_criteria = config.get("inclusion_criteria")

        self.pipeline = Pipeline(
            giv1=self.h['giv1'], siv1=self.h['siv1'], sivv1=self.h['sivv1'],
            giv2=self.h['giv2'], siv2=self.h['siv2'], sivv2=self.h['sivv2'],
            inclusion_criteria=self.inclusion_criteria,
            **config.get("cached"))
        self.columns = ["studyName", "effectSize", "effectSizeSampleSize"]

        env = Environment(loader=FileSystemLoader(config['template_folder']))
        env.filters['custom_enumerate'] = custom_enumerate
        self.template = env.get_template(config["report_template"])

        self.custom_content = {k: v for k, v in config.items() if k.endswith("custom")}

    def get_type_hypothesis(self):
        """ From hypothesis dict, get the type of hypothesis """
        if "mod_val" in self.h:
            return "study_mod"
        if "mod1" in self.h:
            return "var_mod"
        return "regular"

    def get_text_hypothesis(self, templates, hyp):
        """ Input = template to use + hypothesis, Output = text, human-readable hypothesis """
        if self.type_h == "regular":
            return templates["regular"].format(hyp['comparative'], hyp['siv1'],
                hyp['sivv1'], hyp['siv2'], hyp['sivv2'])
        if self.type_h == "numerical_moderator":
            return templates["numerical_moderator"].format(hyp['siv1'], hyp['sivv1'],
                hyp['siv2'], hyp['sivv2'], hyp['comparative'], hyp['mod_qualifier'],
                hyp['mod'])
        # type_h == "categorical_moderator"
        return templates["categorical_moderator"].format(hyp['siv1'], hyp['sivv1'],
            hyp['siv2'], hyp['sivv2'], hyp['mod1'], hyp['mod'], hyp['comparative'],
            hyp['mod2'], hyp['mod'])

    def get_pval_paragraph(self, pval):
        """
        From template: 
        IF {$p} ≤ 0.05 This results indicates that Cooperation is significantly {higher/lower} 
        when {siv1} is {sivv1} compared to when {siv2} is {sivv2}.
        IF {$p} > 0.05 This results indicate no significant difference in cooperation 
        when {siv1} is {sivv1} compared to when {siv2} is {sivv2}.
        """
        if pval <= 0.05:
            return f'This result indicates that "{self.text_h}"'
        return f'This result indicates that "{generate_hypothesis(h=self.h, th=self.type_h)}"'

    @staticmethod
    def get_qep_paragraph(qep, k, qe):
        """
        From template:
        IF $QEp ≤ 0.05: contained more variation than would be expected 
        by chance, Q({$k-1}) = {$QE}, p = {$QEp}.
        IF $QEp > 0.05: did not contain more variation than would be expected 
        by chance, Q({$k-1}) = {$QE}, p = {$QEp}.
        """
        output = "Moreover, the overall effect size distribution "
        if qep <= 0.05:
            output += "contained more variation "
        else:
            output += "did not contain more variation "

        return output + f"than would be expected by chance, \\(Q({k-1}) = " + \
            f"{round(qe, 3)}\\), \\(p = {round(qep, 3)}\\)."

    def generate_html_review(self, **kwargs):
        """ self explanatory """
        html_review = self.template.render(**kwargs)
        return html_review

    def get_figures(self, data):
        """ Get figures to be displayed """
        df_map =data.groupby("country").agg({"observation": "nunique"}).reset_index()
        fig_map = px.choropleth(
            df_map, locations="country",
            color="observation",
            color_continuous_scale=px.colors.sequential.Plasma)

        fig_hist_year = px.histogram(data, x="paperYearDOI", nbins=50)
        return {
            "fig_study_provenance": fig_map.to_html(),
            "fig_study_year": fig_hist_year.to_html()}

    def format_moderator(self):
        """ Format moderator for meta-analysis """
        if self.type_h == 'regular':
            return None
        return {self.h['type_mod']: [self.h['mod']]}

    def get_data_ma(self, output, var):
        """ Retrieves data to display in the meta-analysis """
        columns_info = {
            "studyName": "ID", "citation": "Study", "overallN": "N",
            "country": "Country", "effectSize": self.config["es_measure"],
            "effectSizeLowerLimit": "Lower Bound CI", 
            "effectSizeUpperLimit": "Upper Bound CI", 
        }
        columns = ["studyName", "citation", "overallN", "country"] + \
            var + ["effectSize", "effectSizeLowerLimit", "effectSizeUpperLimit"]
        return output[columns].rename(columns=columns_info)

    def get_variables(self):
        """ Include any moderator + control variables """
        res = []
        if "mod" in self.h:
            res.append(self.h["mod"])
        if "control_variables" in self.config:
            res += self.config["control_variables"]
        return res

    def get_data_variables(self, var):
        """ Retrieves description of variables """
        filtered_df = self.label_des[self.label_des['node'] \
            .apply(lambda x: any(x.endswith(y) for y in var))]
        filtered_df["node"] = filtered_df["node"].apply(lambda x: x.split("#")[-1].split("/")[-1])
        return filtered_df

    def __call__(self, save_folder):
        """ Run meta-analysis and produce meta-review """
        variables = self.get_variables()
        output = self.pipeline(
            data=self.data, es_measure=self.config["es_measure"],
            method=self.config["method_mv"], mods=self.format_moderator(),
            variables=variables)
        type_method = "mixed effects" if self.config["method_mv"] in ["ML", "REML"] \
            else "fixed effects"
        ma_res, refs = output["results_rma"], output["refs"]

        type_es = "standardized mean difference" \
            if self.config["es_measure"] == "d" else "raw correlation coefficient"
        name_es = "Cohen" if self.config["es_measure"] == "d" else "Pearson"
        args = {
            "hypothesis": self.text_h, "name": self.h['giv1'], "info": self.h,
            "type_method": type_method,
            "data_ma_es_nb": format(output["data"].shape[0], ','),
            "data_ma_study_nb": format(output["data"].study.unique().shape[0], ','),
            "data_ma_country_nb": format(output["data"].country.unique().shape[0], ','),
            "data_ma_id": 'data_ma_id',
            "data_ma": self.get_data_ma(output=output["data"], var=variables),
            "data_variables_id": "data_variables_id",
            "data_variables": self.get_data_variables(var=variables),
            "ma_res_call": ma_res["call"], "k": ma_res["k"][-1],
            "es": round(ma_res['b'][-1][0], 2), "categorize_es": categorize_es(ma_res['b'][-1][0]),
            "references": self.references,
            "type_es": type_es, "name_es": name_es,
            "ci_lb": round(ma_res["ci.lb"][-1], 3), "ci_ub": round(ma_res["ci.ub"][-1], 3),
            "df_info_id": "df_info_id", "df_info": ma_res['df_info'],
            "mod": self.h.get('mod'),
            "type_mod": self.type_h.split('_', maxsplit=1)[0],
            "ref_mod": refs[self.h.get('mod')] if self.h.get('mod') else None,
            "pval_paragraph": self.get_pval_paragraph(ma_res["pval"][-1]),
            "T2": round(ma_res['tau2'][-1], 3), "T": round(math.sqrt(ma_res['tau2'][-1]), 3),
            "I2": round(ma_res['I2'][-1], 3),
            "qep_paragraph": self.get_qep_paragraph(
                ma_res['QEp'][-1], ma_res["k"][-1], ma_res['QE'][-1]),
            "inclusion_criteria_2": generate_sent_ic(self.inclusion_criteria)
            }
        args.update(self.config)
        args.update(self.custom_content)
        args.update(self.get_figures(data=output["data"]))

        html_review = self.generate_html_review(**args)
        with open(os.path.join(save_folder, "report.html"), "w", encoding="utf-8") as file:
            file.write(html_review)


@click.command()
@click.argument("config")
@click.argument("save_folder")
def main(config, save_folder):
    """ Main meta review generation """
    meta_review = MetaReview(config=config)
    meta_review(save_folder=save_folder)


if __name__ == '__main__':
    # python src/meta_review.py src/configs/meta_review_regular_h.json meta_review/regular
    main()
