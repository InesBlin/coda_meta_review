# -*- coding: utf-8 -*-
"""
Meta-review generation in Python
"""
import os
import re
import json
import math
from typing import Union, Dict
from copy import deepcopy
import yaml
import click
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import plotly.express as px
from src.pipeline import Pipeline


def custom_enumerate(iterable, start=1):
    """ Custom enumerate for HTML """
    return enumerate(iterable, start)


def load_yaml_file(filename):
    """ Self explanatory """
    with open(filename, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data


def load_json_file(filename):
    """ Self explanatory """
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def categorize_es(x):
    """ categorize effect size """
    if abs(x) >= 0.5:
        return "large"
    if abs(x) >= 0.2:
        return "small"
    return "null"


def categorize_pval(x):
    """ categorize pvalue """
    if x < 0.05:
        return "statistically significant"
    return "not statistically significant"


def replace_cite_id(input_text, dico):
    """ In text, replace ~\\cite{} with [nb] """
    pattern = r'<cite (.*?)>'
    def replace_with_dict(match):
        key = match.group(1)
        return f"[{dico.get(key, '')}]"
    return re.sub(pattern, replace_with_dict, input_text)


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
    1- contains `reg_qualifier` as key
    2- contains `num_qualifier`, `mod_qualifier`, `mod`,  `type_mod` as keys
    3- contains `mod1`, `mod`, `cat_qualifier`, `mod2`, `type_mod` as keys

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

        if "config" in config:
            self.config = load_yaml_file(config["config"])
        else:
            keys_config = [
                "title", "authors", "introduction", "inclusion_criteria_1",
                "inclusion_criteria_2", "inclusion_criteria_3", "es_measure",
                "coding_variables", "method_mv"]
            self.config = {k: config.get(k) for k in keys_config}

        self.references = load_json_file(config["references"])
        self.id_ref_to_nb = {x["id"]: i+1 for i, x in enumerate(self.references)}
        self.config = {k: replace_cite_id(v, self.id_ref_to_nb) \
            if isinstance(v, str) else v for k, v in self.config.items()}
        self.structure = load_yaml_file(config["structure"])
        self.key_qualifier = [x for x in self.h.keys() if x.endswith("qualifier")][0]
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

        self.pipeline = Pipeline(
            giv1=self.h['giv1'], siv1=self.h['siv1'], sivv1=self.h['sivv1'],
            giv2=self.h['giv2'], siv2=self.h['siv2'], sivv2=self.h['sivv2'], **config.get("cached"))
        self.columns = ["studyName", "effectSize", "effectSizeSampleSize"]

        env = Environment(loader=FileSystemLoader(config['template_folder']))
        env.filters['custom_enumerate'] = custom_enumerate
        self.template = env.get_template(config["report_template"])

        self.i2_threshold = 25
        self.tau2_threshold = 0.5
        self.custom_content = {k: v for k, v in config.items() if k.endswith("custom")}

    def get_type_hypothesis(self):
        """ From hypothesis dict, get the type of hypothesis """
        if "reg_qualifier" in self.h:
            return "regular"
        if "num_qualifier" in self.h:
            return "numerical_moderator"
        return "categorical_moderator"

    def get_text_hypothesis(self, templates, hyp):
        """ Input = template to use + hypothesis, Output = text, human-readable hypothesis """
        if self.type_h == "regular":
            return templates["regular"].format(hyp['reg_qualifier'], hyp['siv1'],
                hyp['sivv1'], hyp['siv2'], hyp['sivv2'])
        if self.type_h == "numerical_moderator":
            return templates["numerical_moderator"].format(hyp['siv1'], hyp['sivv1'],
                hyp['siv2'], hyp['sivv2'], hyp['num_qualifier'], hyp['mod_qualifier'],
                hyp['mod'])
        # type_h == "categorical_moderator"
        return templates["categorical_moderator"].format(hyp['siv1'], hyp['sivv1'],
            hyp['siv2'], hyp['sivv2'], hyp['mod1'], hyp['mod'], hyp['cat_qualifier'],
            hyp['mod2'], hyp['mod'])

    def get_comparison_hypothesis(self, type_h):
        """ compare hypothesis to data """
        if type_h == "verified":
            return type_h, f"The hypothesis is verified: {self.text_h}"

        if type_h == "verified_not_sig":
            hyp = self.get_text_hypothesis(templates=self.h_template_not_stat_sig, hyp=self.h)
            return type_h, f"""
                Although the effect size is small to large, the meta-analysis was not statistically significant.
                The original hypothesis was: {self.text_h}
                A more accurate hypothesis would be: {hyp}"""

        if type_h == "not_verified":
            qualifier = "higher" if self.h[self.key_qualifier] == "lower" else "lower"
            hyp = deepcopy(self.h)
            hyp[self.key_qualifier] = qualifier
            return type_h, f"""
                The hypothesis is incorrect.
                The original hypothesis was: {self.text_h}
                The correct hypothesis is: {self.get_text_hypothesis(templates=self.h_template, hyp=hyp)}
                """

        if type_h == "not_verified_not_sig":
            qualifier = "higher" if self.h[self.key_qualifier] == "lower" else "lower"
            hyp = deepcopy(self.h)
            hyp[self.key_qualifier] = qualifier
            hyp = self.get_text_hypothesis(templates=self.h_template_not_stat_sig, hyp=hyp)
            return type_h, f"""
                The hypothesis seems to be incorrect but is not statistically significant.
                The original hypothesis was: {self.text_h}
                A more accurate hypothesis would be: {hyp}
                """

        return type_h, f"""
        The effect size is non-informative. 
        The original hypothesis was: "{self.text_h}"
        This hypothesis is not validated.
        """

    def verify_hypothesis(self, es, pval):
        """ Compare outcome to hypothesis """
        if es > 0.2:
            if self.h[self.key_qualifier] == "higher":
                if pval < 0.05:
                    return self.get_comparison_hypothesis(type_h="verified")
                # pval >= 0.05
                return self.get_comparison_hypothesis(type_h="verified_not_sig")
            # self.h[self.key_qualifier] == 'lower'
            if pval < 0.05:
                return self.get_comparison_hypothesis(type_h="not_verified")
            # pval >= 0.05
            return self.get_comparison_hypothesis(type_h="not_verified_not_sig")

        if es < -0.2:
            if self.h[self.key_qualifier] == "lower":
                if pval < 0.05:
                    return self.get_comparison_hypothesis(type_h="verified")
                # pval >= 0.05
                return self.get_comparison_hypothesis(type_h="verified_not_sig")
            # self.h[self.key_qualifier] == 'lower'
            if pval < 0.05:
                return self.get_comparison_hypothesis(type_h="not_verified")
            # pval >= 0.05
            return self.get_comparison_hypothesis(type_h="not_verified_not_sig")
        # -0.2<=es<=0.2 -> null finding
        return self.get_comparison_hypothesis(type_h="inconclusive")

    def generate_html_review(self, **kwargs):
        """ self explanatory """
        html_review = self.template.render(**kwargs)
        return html_review

    def get_figures(self, data):
        """ Get figures to be displayed """
        df_map =data.groupby("country").agg({"observation": "nunique"}).reset_index()
        fig_map = px.choropleth(
            df_map, locations="country",
            color="observation", # lifeExp is a column of gapminder
            # hover_name="country", # column to add to hover information
            color_continuous_scale=px.colors.sequential.Plasma)

        fig_hist_year = px.histogram(data, x="paperYearDOI", nbins=50)
        return {
            "fig_study_provenance": fig_map.to_html(),
            "fig_study_year": fig_hist_year.to_html()}

    def get_i2_description(self, i2):
        """ Describe heterogeneity across studies """
        if i2 >= self.i2_threshold:
            return f"There is a high heterogeneity across studies (I2 = {round(i2, 2)}%), " + \
                "which means that the studies cannot be considered studies of the same " + \
                    "population. We suggest that a finer-grained subgroup analysis might " + \
                        "be worthwhile."
        return f"There is a low heterogeneity across studies (I2 = {round(i2, 2)}%), " + \
            "which means that the studies can be considered studies of the same population. "

    def get_tau2_description(self, tau2):
        """ Describe dispersion of time effect sizes """
        how = "low" if tau2 <= self.tau2_threshold else "high"
        return f"There is a {how} dispersion of time effect sizes " + \
            f"(tau = {round(math.sqrt(tau2), 2)}, T2 = {round(tau2, 2)})."

    def format_moderator(self):
        """ Format moderator for meta-analysis """
        if self.type_h == 'regular':
            return None
        return {self.h['type_mod']: [self.h['mod']]}

    def __call__(self, save_folder):
        """ Run meta-analysis and produce meta-review """
        output = self.pipeline(
            data=self.data, es_measure=self.config["es_measure"],
            method=self.config["method_mv"], mods=self.format_moderator())
        ma_res, refs = output["results_rma"], output["refs"]
        es, pval, i2 = ma_res['b'][-1][0], ma_res["pval"][-1], ma_res['I2'][-1]
        ci_lb, ci_ub = ma_res["ci.lb"][-1], ma_res["ci.ub"][-1]

        type_es = "standardized mean difference" \
            if self.config["es_measure"] == "d" else "raw correlation coefficient"
        name_es = "Cohen" if self.config["es_measure"] == "d" else "Pearson"
        ex_mod_read = [x for x in ma_res['df_info'].info_treatment.values if x != 'intrcpt'][0]
        type_h, conclude_h = self.verify_hypothesis(es=es, pval=pval)

        args = {
            "hypothesis": self.text_h, "name": self.h['giv1'], "info": self.h,
            "data_ma_shape": format(output["data"].shape[0], ','),
            "data_ma_es_nb": format(output["data"].shape[0], ','),
            "data_ma_study_nb": format(output["data"].study.unique().shape[0], ','),
            "data_ma_country_nb": format(output["data"].country.unique().shape[0], ','),
            "columns": self.columns, "data_ma_id": 'data_ma_id',
            "data_ma": output["data"][self.columns], "ma_res_call": ma_res["call"],
            "es": round(es, 2), "categorize_es": categorize_es(es),
            "pval": round(pval, 3), "categorize_pval": categorize_pval(pval),
            "conclude_hypothesis": conclude_h,
            "references": self.references,
            "type_es": type_es, "name_es": name_es,
            "i2_description": self.get_i2_description(i2=i2),
            "tau2_description": self.get_tau2_description(tau2=ma_res['tau2'][-1]),
            "ci_lb": round(ci_lb, 3), "ci_ub": round(ci_ub, 3),
            "df_info_id": "df_info_id", "df_info": ma_res['df_info'],
            "type_h": self.type_h, "mod": self.h.get('mod'),
            "type_mod": self.type_h.split('_', maxsplit=1)[0],
            "ref_mod": refs[self.h.get('mod')] if self.h.get('mod') else None,
            "ex_mod_read": ex_mod_read,
            "ex_mod_val": ex_mod_read.replace(self.h.get('mod'), '') if self.h.get('mod') else None
            }
        args.update(self.config)
        args.update(self.structure)
        args.update(self.custom_content)
        args.update(self.get_figures(data=output["data"]))

        html_review = self.generate_html_review(**args)
        with open(os.path.join(save_folder, "report.html"), "w", encoding="utf-8") as file:
            file.write(html_review)
        with open(os.path.join(save_folder, "type_h.txt"), "w", encoding="utf-8") as file:
            file.write(type_h)


@click.command()
@click.argument("config")
@click.argument("save_folder")
def main(config, save_folder):
    """ Main meta review generation """
    meta_review = MetaReview(config=config)
    meta_review(save_folder=save_folder)


if __name__ == '__main__':
    # python src/meta_review.py src/configs/meta_review_regular_h.json meta_review/regular
    # python src/meta_review.py src/configs/meta_review_categorical_h.json meta_review/categorical
    # python src/meta_review.py src/configs/meta_review_numerical_h.json meta_review/numerical
    main()
