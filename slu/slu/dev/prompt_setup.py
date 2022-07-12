"""
Generate config for prompts using nls-keys downloaded from studio

Usage:
    prompt_setup.py

Options:
    -h --help   Show this screen.
    --version   Show version.
"""

import os
import argparse
import json
import yaml
from typing import List

import pandas as pd
import re
import string
from tqdm import tqdm

from slu import constants as const
from slu.utils import logger

def preprocess_prompt(prompt: str, remove_var: bool = True, fill_token: str = "<pad>") -> str:

    """
    REGEX1: Detect special characters
    REGEX2: Detect noisy digits, alphanumberic characters
    REGEX3: Detect consequtive black spaces
    REGEX4: Detect variables defined inside prompts. Eg: {{.variable}} 
    """

    REGEX1 = re.compile("\[.*?\]")
    REGEX2 = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    REGEX3 = re.compile("/\W/g")
    REGEX4 = re.compile(r'{{(.*?)}}')

    if not isinstance(prompt,str):
        return prompt
        
    prompt = prompt.lower()
    prompt = re.sub(REGEX1, "", prompt)
    prompt = re.sub(REGEX2, "", prompt)
    prompt = re.sub(REGEX3, "", prompt)
    prompt = re.sub(' +', ' ', prompt)
    prompt = re.sub('_', " ", prompt)
    prompt = prompt.strip()
    if remove_var:
        prompt = re.sub(REGEX4, fill_token ,prompt)
    prompt = prompt.translate(str.maketrans("", "", string.punctuation.replace("<","").replace(">","")))
    
    return prompt


def _valid_string(string: str) -> bool:
    if isinstance(string, str):
        if all([len(string) > 0, string != 'nan', string != '', string != " ", "Unnamed" not in string]):
            return True
    return False


def _nls_to_state(string: str) -> str:
    """
    Convert NLS-Key to State. 
    NLS-Keys are nothing but extentions of State names. 
    Eg: state "A" will have NLS-Keys names A_1, A_2, and so on.
    This code simply removes the suffix _1, _2, etc from a NLS-Key to get the original State name. 
    """
    if not _valid_string(string):
        return None
    string = string.split()
    if len(string) > 1:
        string = "_".join(_ for _ in string[:len(string)-1])
    return string


def get_prompts_map(df: pd.DataFrame) -> pd.DataFrame:

    prompts_map: dict = dict()
    supported_languages: list = [_ for _ in df.columns if (_valid_string(_) and _ not in [const.NLS_KEY,const.STATE])]

    if not supported_languages:
        raise Exception(
            "No languages found in the dataset"
        )

    logger.debug(f"Found languages: {supported_languages}")
    for lang in supported_languages:
        prompts_map[lang] = {}

        for i in tqdm(range(df.shape[0]),desc = f"Fetching prompts for {lang}"):
            state = df.iloc[i][const.STATE]
            prompt =  df.iloc[i][lang]

            if _valid_string(prompt):
                prompt = preprocess_prompt(prompt, fill_token=const.PROMPT_NOISE_FILLER_TOKEN)
                # logger.debug(prompt)

            if not state in prompts_map[lang]:
                prompts_map[lang][state] = []
            
            prompts_map[lang][state].append(prompt)

    return prompts_map


def validate(df: pd.DataFrame) -> pd.DataFrame:
    
    if not const.NLS_KEY in df.columns:
        raise Exception(
            f"Mandatory column missing {const.NLS_KEY}"
        )
    if not const.STATE in df.columns:
        raise Exception(
            f"Mandatory column missing {const.STATE}"
        )

    df = df[(df[const.NLS_KEY].notna()) & (df[const.STATE].notna())]
    invalid_rows = df[(df[const.NLS_KEY].isna()) | (df[const.STATE].isna())]
    logger.debug(f"Num invalid rows: {invalid_rows.shape[0]}")


def setup_prompts(args: argparse.Namespace) -> None:
    
    dataset: str = args.file
    overwrite: bool = args.overwrite or True
    dest: str = args.dest or const.PROMPTS_CONFIG_PATH

    if os.path.exists(dest) and not overwrite:
        raise RuntimeError(
            f"""
            File already exists {dest}.
            Use --overwrite=True
            """.strip()
        )

    data_frame = pd.read_csv(dataset)
    if (const.STATE not in data_frame.columns and const.NLS_KEY in data_frame.columns):
            logger.debug(f"State column missing, deriving from NLS-Keys")
            data_frame[const.STATE] = data_frame[const.NLS_KEY].apply(lambda x: _nls_to_state(x))

    validate(data_frame)
    prompts_map = get_prompts_map(data_frame)

    with open(os.path.join(dest), 'w') as file:
        yaml.safe_dump(prompts_map, file, allow_unicode=True)