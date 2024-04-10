import os
import re
import pandas as pd
import numpy as np
from copy import deepcopy
from src.pipeline import Pipeline
from jinja2 import Environment, FileSystemLoader
import plotly.express as px

def categorize_es(x):
    """ categorize effect size """
    if abs(x) >= 0.5:
        return "large"
    if abs(x) >= 0.2:
        return "small"
    return "null"

def categorize_pval(x):
    """ categorize effect size """
    if x < 0.05:
        return "statistically significant"
    return "not statistically significant"

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
    2- contains `num_qualifier`, `mod_qualifier`, `mod` as keys
    3- contains `mod1`, `mod`, `cat_qualifier`, `mod2`as keys

    [NB]: for now focus on 1-
    """

    def __init__(self, **args):
        """ Init main templates/variables """
        self.data = args["data"]
        self.h = args["hypothesis"]
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
            giv2=self.h['giv2'], siv2=self.h['siv2'], sivv2=self.h['sivv2'], **args.get("cached"))
        self.columns = ["studyName", "effectSize", "effectSizeSampleSize"]

        env = Environment(loader=FileSystemLoader('.'))
        self.template = env.get_template(args["report_template"])

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
            return f"The hypothesis is verified: {self.text_h}"

        if type_h == "verified_not_sig":
            hyp = self.get_text_hypothesis(templates=self.h_template_not_stat_sig, hyp=self.h)
            return f"""
                Although the effect size is small to large, the meta-analysis was not statistically significant.
                The original hypothesis was: {self.text_h}
                A more accurate hypothesis would be: {hyp}"""

        if type_h == "not_verified":
            qualifier = "higher" if self.h[self.key_qualifier] == "lower" else "lower"
            hyp = deepcopy(self.h)
            hyp[self.key_qualifier] = qualifier
            return f"""
                The hypothesis is incorrect.
                The original hypothesis was: {self.text_h}
                The correct hypothesis is: {self.get_text_hypothesis(templates=self.h_template, hyp=hyp)}
                """

        if type_h == "not_verified_not_sig":
            qualifier = "higher" if self.h[self.key_qualifier] == "lower" else "lower"
            hyp = deepcopy(self.h)
            hyp[self.key_qualifier] = qualifier
            hyp = self.get_text_hypothesis(templates=self.h_template_not_stat_sig, hyp=hyp)
            return f"""
                The hypothesis seems to be incorrect but is not statistically significant.
                The original hypothesis was: {self.text_h}
                A more accurate hypothesis would be: {hyp}
                """

        return f"""
        The effect size is non-informative. 
        The original hypothesis was: {self.text_h}.
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
    
    def __call__(self, save_folder):
        """ Run meta-analysis and produce meta-review """
        data_ma = self.pipeline.get_data_meta_analysis(self.data)
        data_ma.to_csv(os.path.join(save_folder, "data_ma_init.csv"))
        output = self.pipeline(data=self.data)
        ma_res, refs = output["results_rma"], output["refs"]
        output["data"].to_csv(os.path.join(save_folder, "data_ma_pre_ma.csv"))
        es, pval = ma_res['b'][0][0], ma_res["pval"][0]

        df_map = data_ma.groupby("country").agg({"observation": "nunique"}).reset_index()
        fig_map = px.choropleth(
            df_map, locations="country",
            color="observation", # lifeExp is a column of gapminder
            # hover_name="country", # column to add to hover information
            color_continuous_scale=px.colors.sequential.Plasma)
        fig_map.write_html(os.path.join(save_folder, "fig_map.html"))

        fig_hist_year = px.histogram(data_ma, x="paperYearDOI", nbins=50)
        fig_hist_year.write_html(os.path.join(save_folder, "fig_hist_year.html"))

        conclude_hypothesis = self.verify_hypothesis(es=es, pval=pval)
        title = "Meta-Review"

        args = dict(
            hypothesis=self.text_h, name=self.h['giv1'], info=self.h,
            data_ma_shape=format(data_ma.shape[0], ','),
            columns=self.columns, data_ma=data_ma[self.columns], ma_res_call=ma_res["call"],
            es=round(es, 2), categorize_es=categorize_es(es), title=title,
            pval=pval, categorize_pval=categorize_pval(pval),
            conclude_hypothesis=conclude_hypothesis)

        html_review = self.generate_html_review(**args)
        with open(os.path.join(save_folder, "report.html"), "w", encoding="utf-8") as file:
            file.write(html_review)

ARGS = {
    "data": pd.read_csv("./data/observationData.csv", index_col=0),
    "cached": {
        "study_moderators": "./data/moderators/study_moderators.csv",
        "country_moderators": "./data/moderators/country_moderators.csv",
        "simple_country_moderators": "./data/moderators/simple_country_moderators.csv",
        "complex_country_moderators": "./data/moderators/complex_country_moderators.csv",
        "variable_moderators": "./data/moderators/variable_moderators.csv"
    },
    "report_template": 'meta_review/report_template.html',
    "hypothesis": {
        "giv1": "group_size", "siv1": "decision maker", "sivv1": "individual",
        "giv2": "group_size", "siv2": "group size level", "sivv2": "high",
        'reg_qualifier': 'higher'
    }
}


if __name__ == '__main__':
    META_REVIEW = MetaReview(**ARGS)
    META_REVIEW(save_folder="meta_review")

