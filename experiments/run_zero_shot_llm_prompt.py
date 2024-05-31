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

MODEL_3 = "gpt-3.5-turbo-0125"
MODEL_4 = "gpt-4o"

def write_readable_h(output_f, df_output, llmhg):
    """ Save human readable hypotheses in txt file """
    f_readable = open(output_f, 'w', encoding='utf-8')
    for _, row in df_output.iterrows():
        f_readable.write(f"{llmhg.generate_readable_h(row=row)}\n")
    f_readable.close()

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
    nb_prompt = 0
    create_if_not_exists(folder_out)
    curr_time = str(datetime.now())
    f_errors = open(os.path.join(folder_out, f"errors_{curr_time[:10]}_{curr_time[11:19]}.txt"), "w+", encoding='utf-8')
    for th in TYPE_HYPOTHESIS:
        for esm in ES_MEASURE:
            logger.info(f"Prompting for hypothesis `{th}` with effect size measure `{esm}`")
            folder = os.path.join(folder_out, f"h_{th}_es_{esm}")
            for f in [folder, os.path.join(folder, "prompts"), os.path.join(folder, "outputs")]:
                create_if_not_exists(folder=f)
            data = pd.read_csv(os.path.join(folder_in, f"h_{th}_es_{esm}.csv"), index_col=0)

            for giv in tqdm(data.giv_prop.unique()):
                name = giv.split('/')[-1]
                output_f = os.path.join(folder, "outputs", f"{name}_readable.txt")
                nb_prompt += 1
                if not os.path.exists(output_f):
                    df = data[data.giv_prop == giv]
                    df.to_csv(os.path.join(folder, "prompts", f"{name}.csv"))
                    data_prompt = format_data_for_prompt(df=df)

                    
                    try:
                        try:  # First trying with GPT3 is the context window is enough
                            llmhg = LLMHypothesisGeneration(type_hypothesis=th, model=MODEL_3)
                            output = llmhg(data=data_prompt)
                            df_output = output.replace("```csv", "").replace("```", "")
                            df_output = pd.read_csv(StringIO(df_output))
                            df_output.to_csv(os.path.join(folder, "outputs", f"{name}_csv.csv"))
                            write_readable_h(output_f, df_output, llmhg)
                        except:  # if context window too big -> switching models
                            llmhg = LLMHypothesisGeneration(type_hypothesis=th, model=MODEL_4)
                            output = llmhg(data=data_prompt)
                            df_output = output.replace("```csv", "").replace("```", "")
                            df_output = pd.read_csv(StringIO(df_output))
                            df_output.to_csv(os.path.join(folder, "outputs", f"{name}_csv.csv"))
                            write_readable_h(output_f, df_output, llmhg)
                    
                    except Exception as e:
                        f_errors.write(f"Log error for hypothesis `{th}` with effect size measure `{esm}`\n")
                        f_errors.write(f"GIV: {giv}\t ({name})\n")
                        f_errors.write(str(e)+"\n=========\n\n")
                    
                    # Else doing with
    
    f_errors.close()
    print(f"# of prompts: {nb_prompt}")


if __name__ == '__main__':
    # python experiments/run_zero_shot_llm_prompt.py data/hypotheses/llm experiments/llm_zero_shot_prompting
    main()
