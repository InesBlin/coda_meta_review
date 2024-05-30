# -*- coding: utf-8 -*-
"""
Running zero-shot prompting with LLM for automated hypothesis generation
"""
import os
import click
from io import StringIO
from tqdm import tqdm
from datetime import datetime
from loguru import logger
import pandas as pd
from src.lp.llm import LLMHypothesisGeneration

def create_if_not_exists(folder: str):
    """ Create folder directory if doesn't already exist """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
def format_data_for_prompt(df: pd.DataFrame) -> str:
    filter_out_cols = ["giv_prop", "obs"]
    df = df[[x for x in df.columns if x not in filter_out_cols]]
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.split("/")[-1])
    # Random shuffling
    df = df.sample(frac=1).reset_index(drop=True)
    return f"```csv\n{df.to_csv(index=False)}```"
    


TYPE_HYPOTHESIS = ['regular', 'var_mod', 'study_mod']
ES_MEASURE = ['r', 'd']

@click.command()
@click.argument("folder_in")
@click.argument("folder_out")
def main(folder_in, folder_out):
    """ Prompting for top 5 hypothesis for 
    - each giv of data
    - each type of hypothesis """
    create_if_not_exists(folder_out)
    curr_time = str(datetime.now())
    f_errors = open(os.path.join(folder_out, f"errors_{curr_time[:10]}_{curr_time[11:19]}.txt"), "w+", encoding='utf-8')
    for th in TYPE_HYPOTHESIS:
        llmhg = LLMHypothesisGeneration(type_hypothesis=th)
        for esm in ES_MEASURE:
            logger.info(f"Prompting for hypothesis `{th}` with effect size measure `{esm}`")
            folder = os.path.join(folder_out, f"h_{th}_es_{esm}")
            for f in [folder, os.path.join(folder, "prompts"), os.path.join(folder, "outputs")]:
                create_if_not_exists(folder=f)
            data = pd.read_csv(os.path.join(folder_in, f"h_{th}_es_{esm}.csv"), index_col=0)

            for giv in tqdm(data.giv_prop.unique()):
                name = giv.split('/')[-1]
                input_f = os.path.join(folder, "prompts", f"{name}.csv")
                if not os.path.exists(input_f):
                    df = data[data.giv_prop == giv]
                    df.to_csv(input_f)
                    data_prompt = format_data_for_prompt(df=df)

                    try:
                        output = llmhg(data=data_prompt)
                        df_output = output.replace("```csv", "").replace("```", "")
                        df_output = pd.read_csv(StringIO(df_output))
                        df_output.to_csv(os.path.join(folder, "outputs", f"{name}_csv.csv"))

                        f_readable = open(os.path.join(folder, "outputs", f"{name}_readable.txt"), 'w', encoding='utf-8')
                        for _, row in df_output.iterrows():
                            f_readable.write(f"{llmhg.generate_readable_h(row=row)}\n")
                        f_readable.close()
                    
                    except Exception as e:
                        f_errors.write(f"Log error for hypothesis `{th}` with effect size measure `{esm}`\n")
                        f_errors.write(f"GIV: {giv}\t ({name})\n")
                        f_errors.write(str(e)+"\n=========\n\n")
    
    f_errors.close()


if __name__ == '__main__':
    # python experiments/run_zero_shot_llm_prompt.py data/hypotheses/llm data/outputs/llm
    main()
