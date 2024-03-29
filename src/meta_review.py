import re
import pandas as pd
import numpy as np
from src.pipeline import Pipeline
from jinja2 import Environment, FileSystemLoader
import plotly.express as px

DATA = pd.read_csv("./data/observationData.csv", index_col=0)
REGEX_T1_VS_T2 = r"Cooperation is significantly (higher|lower) when (.*) is (.*) compared to when (.* )is (.*)\."
NAME = "group_size"
HYPOTHESIS = "Cooperation is significantly higher when decision maker is individual compared to when group size level is high."

CACHED = {
        "study_moderators": "./data/moderators/study_moderators.csv",
        "country_moderators": "./data/moderators/country_moderators.csv",
        "simple_country_moderators": "./data/moderators/simple_country_moderators.csv",
        "complex_country_moderators": "./data/moderators/complex_country_moderators.csv",
        "variable_moderators": "./data/moderators/variable_moderators.csv"
    }

def get_param_info_one_line(text):
    """ Retrieve params with regex """
    matches = re.finditer(REGEX_T1_VS_T2, text, re.MULTILINE)
    for _, match in enumerate(matches, start=1):
        return match.groups()
    return None

def format_info(info, name):
    """ List to dict """
    return {"giv1": name, "siv1": info[1].strip(), "sivv1": info[2].strip(),
            "giv2": name, "siv2": info[3].strip(), "sivv2": info[4].strip(),
            "outcome": info[0].strip()}

def categorize_es(x):
    """ categorize effect size """
    if abs(x) >= 0.5:
        return "large"
    if abs(x) >= 0.2:
        return "small"
    return "null"

INFO = get_param_info_one_line(text=HYPOTHESIS)
INFO = format_info(info=INFO, name=NAME)

PIPELINE = Pipeline(
    giv1=INFO['giv1'], siv1=INFO['siv1'], sivv1=INFO['sivv1'],
    giv2=INFO['giv2'], siv2=INFO['siv2'], sivv2=INFO['sivv2'], **CACHED)
DATA_MA = PIPELINE.get_data_meta_analysis(data=DATA)
COLUMNS = ["studyName", "effectSize", "effectSizeSampleSize"]
MA_RES, REFS = PIPELINE(data=DATA)
ES = MA_RES['b'][0][0]

DF_MAP = DATA_MA.groupby("country").agg({"observation": "nunique"}).reset_index()
FIG_MAP = px.choropleth(
    DF_MAP, locations="country",
    color="observation", # lifeExp is a column of gapminder
    # hover_name="country", # column to add to hover information
    color_continuous_scale=px.colors.sequential.Plasma)
FIG_MAP.write_html("meta_review/fig_map.html")

def generate_html_review(**kwargs):
    # Load Jinja2 environment and template
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('meta_review/report_template.html')

    # Render template with provided data
    html_review = template.render(**kwargs)

    return html_review

TITLE = "Meta-Review"
ARGS = dict(
    hypothesis=HYPOTHESIS, name=NAME, info=INFO,
    data_ma_shape=format(DATA_MA.shape[0], ','),
    columns=COLUMNS, data_ma=DATA_MA[COLUMNS], ma_res_call=MA_RES["call"],
    es=round(ES, 2), categorize_es=categorize_es(ES), title=TITLE)

HTML_REVIEW = generate_html_review(**ARGS)
with open("meta_review/report.html", "w", encoding="utf-8") as file:
    file.write(HTML_REVIEW)


# def generate_review():
#     print(f"Hypothesis explored: {HYPOTHESIS}")
#     print(f"Treatment 1: GIV = {NAME}, SIV = {INFO['siv1']}, SIVV = {INFO['sivv1']}")
#     print(f"Treatment 2: GIV = {NAME}, SIV = {INFO['siv2']}, SIVV = {INFO['sivv2']}")
#     print(f"How many observations are retrieve? In total, they are {format(DATA_MA.shape[0], ',')}.")
#     print("Table 1: Effect Size Annotation")
#     print(DATA_MA[COLUMNS])
#     print("Run a meta-analysis")
#     print(MA_RES["call"])
#     print(f"The overall meta-analytic effect size is {round(ES, 2)}. This effect can be considered {categorize_es(ES)}.")

    
    

# generate_review()


# # Generate sample data
# data = {
#     'A': np.random.randn(10),
#     'B': np.random.rand(10) * 100,
#     'C': np.random.choice(['X', 'Y', 'Z'], 10)
# }
# df = pd.DataFrame(data)

# # Render DataFrame to HTML
# df_html = df.to_html(index=False)

# # Prepare Jinja2 environment
# env = Environment(loader=FileSystemLoader('.'))
# template = env.get_template('report_template.html')

# # Render template
# output = template.render(df=df)

# # Save HTML report
# with open('report.html', 'w') as f:
#     f.write(output)
