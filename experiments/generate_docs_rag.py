# -*- coding: utf-8 -*-
"""
Generate text documents from csv
"""
import os
import math
import click
from tqdm import tqdm
import pandas as pd

ES_MEASURES = ["d"]
LABELS = ["regular", "var_mod", "study_mod"]

TEMPLATES_S = [
    ("obs", "type", "Observation"),
    ("obs", "eSmeasure", "d")
]

TEMPLATES_S_O = [
    ("obs", "dependentVariable", "dependent"),
    ("obs", "eSEstimate", "ES"),
    ("obs", "effectSizeSampleSize", "N"), 
    ("obs", "hasEffect", "effect"),
    ("t1", "hasIndependentVariable", "iv"),
    ("t1", "hasModerator", "mod"),
    ("t2", "hasIndependentVariable", "iv"),
    ("t2", "hasModerator", "mod"),
    ("obs", "treatment", "t1"),
    ("obs", "treatment", "t2"),
    ("study", "reportsEffect", "obs"),
    ("t1", "betweenOrWithinParticipantsDesign", "design"),
    ("t1", "nCondition", "n1"),
    ("t2", "nCondition", "n2"),
    ("t1", "sDforCondition", "sd1"),
    ("t2", "sDforCondition", "sd2"),
    ("obs", "effectSizeLowerLimit", "ESLower"),
    ("obs", "effectSizeUpperLimit", "ESUpper"),
    ("paper", "study", "study"),
    ("paper", "doi", "doi"),
    ###
    ("iv", "subPropertyOf", "giv_prop"),
    ("mod", "subPropertyOf", "giv_prop"),
    ("iv", "range", "range_class_iv"),
    ("range_class_iv", "subClassOf", "range_superclass_iv"),
    ("range_superclass_iv", "subClassOf", "IndependentVariable"),
    ("mod", "range", "range_class_mod"),
    ("range_class_mod", "subClassOf", "range_superclass_mod"),
    ("range_superclass_mod", "subClassOf", "IndependentVariable")
]

TEMPLATES_S_P_O = [
    ("t1", "iv", "cat_t1"),
    ("t2", "iv", "cat_t2"),
    ("t1", "mod", "mod_t1"),
    ("t2", "mod", "mod_t2"),
    ("study", "mod", "mod_val")
]

def type_of_effect(row):
    """ Categorize effect based on its signifiance """
    if math.isnan(row.ESLower) or math.isnan(row.ESUpper):
        if row.ES > -0.2 and row.ES < 0.2:
            return 'noEffect'
        return 'positive' if row.ES >= 0.2 else 'negative'
    if row.ESLower <= 0 <= row.ESUpper:
        return 'noEffect'
    return 'positive'  if float(row.ES) > 0 else 'negative'



def generate_text(row):
    """ Generate text from row """
    text = []
    for s, p, o in TEMPLATES_S:
        if s in row.index and row[s]:
            text.append(f"{row[s]} {p} {o}")
    for s, p, o in TEMPLATES_S_O:
        if (s in row.index) and row[s] and (o in row.index) and row[o]:
            text.append(f"{row[s]} {p} {row[o]}")
    for s, p, o in TEMPLATES_S_P_O:
        if (s in row.index) and row[s] and (o in row.index) and row[o] \
            and (p in row.index) and row[p]:
            text.append(f"{row[s]} {row[p]} {row[o]}")
    return "\n".join(text)

def generate_docs(df: pd.DataFrame, save_folder: str):
    """ Generate document from csv data """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        save_path = os.path.join(save_folder, f"{index}.txt")
        if not os.path.exists(save_path):
            text = generate_text(row)
            f = open(save_path, "w", encoding="utf-8")
            f.write(text)
            f.close()


@click.command()
@click.argument("folder_in")
@click.argument("folder_out")
def main(folder_in, folder_out):
    """ Saving .txt documents for RAG """
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    for th in LABELS:
        for es in ES_MEASURES:
            save_folder = os.path.join(folder_out, f"h_{th}_es_{es}")
            print(save_folder)
            df = pd.read_csv(os.path.join(folder_in, f"h_{th}_es_{es}.csv"))
            tqdm.pandas()
            df["effect"] = df.progress_apply(type_of_effect, axis=1)
            df = df[df.effect != "noEffect"]
            generate_docs(df=df, save_folder=save_folder)




if __name__ == '__main__':
    # python experiments/generate_docs_rag.py data/hypotheses/entry/ data/rag/documents
    main()


