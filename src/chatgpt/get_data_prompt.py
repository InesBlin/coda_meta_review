# -*- coding: utf-8 -*-
"""
Generate prompts based on the data 
"""
import os
import click
from tqdm import tqdm
import pandas as pd

SPARQL_ENDPOINT = "http://localhost:7200/repositories/coda"

PROMPT_TEMPLATE = """
In this task, we are interested in formulating hypotheses on how humans cooperate, and particularly related to <generic_variable> variables. There are constraints on how you can generate the hypotheses. 

[DATA][csv format]
```
siv1,sivv1,siv2,sivv2
<DATA>
```

[HYPOTHESES]
A valid hypothesis should be in the format
```
Cooperation is significantly <higher/lower> when `siv1` is `sivv1`, compared to when `siv2` is `sivv2`
```
or 
```
Cooperation is significantly <higher/lower> when `siv2` is `sivv2`, compared to when `siv1` is `sivv1`
```

[TASK]
You need to generate the most sensible and coherent hypotheses. When generating the hypothesis, keep the bracket and the original names from the csv. In your response, only list the hypotheses.
"""

def write_data(df_input):
    """ Write data in csv format """
    res = []
    for _, row in df_input.iterrows():
        res.append(f"{row.siv1},{row.sivv1},{row.siv2},{row.sivv2}")
    return "\n".join(res)

@click.command()
@click.argument("data_path")
@click.argument("save_folder")
def main(data_path, save_folder):
    """ Store prompt for each generic variable """
    df = pd.read_csv(data_path, index_col=0)
    for gv in tqdm(df.generic1.unique()):
        gv_name = gv.split("/")[-1].replace("Variable", "").lower()
        if f"{gv_name}.txt" not in os.listdir(save_folder):
            data = write_data(df_input=df[df.generic1 == gv])
            prompt = PROMPT_TEMPLATE.replace("<DATA>", data).replace("<generic_variable>", gv_name)
            with open(os.path.join(save_folder, f"{gv_name}.txt"), "w+", encoding="utf-8") as file:
                file.write(prompt)


if __name__ == '__main__':
    main()