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

lang_map = const.NLS_LANG_MAPPING

def preprocess_prompt(prompt: str, remove_var: bool = True, fill_token: str = const.PROMPT_NOISE_FILLER_TOKEN) -> str:

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
        if all([len(string) > 0, string != 'nan', string != '.nan', string != '', string != " ", "Unnamed" not in string]):
            return True
    return False


def _nls_to_state(string: str, delimiter: str = "_") -> str:
    """
    Convert NLS-Key to State. 
    NLS-Keys are nothing but extentions of State names. 
    Eg: state "A" will have NLS-Keys names A_1, A_2, and so on.
    This code simply removes the suffix _1, _2, etc from a NLS-Key to get the original State name. 
    """
    if not _valid_string(string):
        return None
    string = string.split(delimiter)
    if len(string) > 1:
        string = str(delimiter).join(_ for _ in string[:len(string)-1])
    if isinstance(string, list) and len(string) > 0:
        string = string[0]
    
    return string


def _nls_to_df(dataset: str)-> pd.DataFrame:
    nls_labels = None
    nls_keys = set()

    if not dataset.endswith(".yaml"):
        raise RuntimeError(
            f"""
            Invalid extension, .yaml file expected but instead received {dataset}.
            """.strip()
        )

    with open(dataset) as file:
        nls_labels = yaml.load(file, Loader=yaml.FullLoader)
    if not nls_labels:
        raise RuntimeError(
            f"""
            Invalid or empty input file {dataset}.
            """.strip()
        )   
    
    for lang in lang_map.keys():
        if lang_map[lang] not in nls_labels:
            raise Exception(
                f"{lang} not found in nls labels, please check your input file."
            ) 
        if not nls_labels[lang_map[lang]]:
            raise Exception(
                f"No nls-keys found for {lang}, please check your input file."
            )
        
        for _ in nls_labels[lang_map[lang]].keys():
            nls_keys.add(_)

    logger.debug(f"Total unique nls-keys: {len(nls_keys)}")

    nls_df = pd.DataFrame(columns=[const.NLS_LABEL] + list(lang_map.keys()))
    nls_df[const.NLS_LABEL] = pd.Series(list(nls_keys))

    for i in tqdm(range(nls_df.shape[0]),desc="Fetching prompts"):
        NLS_LABEL = nls_df.iloc[i][const.NLS_LABEL]
        for lang in lang_map.keys():
            if NLS_LABEL not in nls_labels[lang_map[lang]]:
                logger.debug(
                    f"nls-key  {NLS_LABEL} not found for lang {lang}"
                )                
            elif not nls_labels[lang_map[lang]][NLS_LABEL]:
                logger.debug(
                    f"Prompt not found for lang {lang}, nls-key {NLS_LABEL}"
                )
            else:
                if isinstance(nls_labels[lang_map[lang]][NLS_LABEL], str) and len(nls_labels[lang_map[lang]][NLS_LABEL]) > 0:
                    nls_df.at[i,lang] = nls_labels[lang_map[lang]][NLS_LABEL]
                if isinstance(nls_labels[lang_map[lang]][NLS_LABEL], list) and len(nls_labels[lang_map[lang]][NLS_LABEL]) == 1:
                    nls_df.at[i,lang] = nls_labels[lang_map[lang]][NLS_LABEL][0]

    return nls_df

def validate(df: pd.DataFrame) -> pd.DataFrame:
    
    if not const.NLS_LABEL in df.columns:
        raise Exception(
            f"Mandatory column missing {const.NLS_LABEL}"
        )
    if not const.STATE in df.columns:
        raise Exception(
            f"Mandatory column missing {const.STATE}"
        )

    df = df[(df[const.NLS_LABEL].notna()) & (df[const.STATE].notna())]
    invalid_rows = df[(df[const.NLS_LABEL].isna()) | (df[const.STATE].isna())]
    logger.debug(f"Num invalid rows: {invalid_rows.shape[0]}")


def get_prompts_map(df: pd.DataFrame) -> pd.DataFrame:

    prompts_map: dict = dict()
    missing_prompts_map: dict = dict()
    supported_languages: list = [_ for _ in df.columns if (_valid_string(_) and _ not in [const.NLS_LABEL,const.STATE])]
    nls_labels: set = set()

    if not supported_languages:
        raise Exception(
            "No languages found in the dataset"
        )

    logger.debug(f"Found languages: {supported_languages}")
    for lang in supported_languages:
        prompts_map[lang] = {}
        missing_prompts_map[lang] = []

        for i in tqdm(range(df.shape[0]),desc = f"Fetching prompts for {lang}"):
            nls_label = df.iloc[i][const.NLS_LABEL]
            nls_labels.add(nls_label)
            prompt =  df.iloc[i][lang]

            if _valid_string(prompt):
                prompt = preprocess_prompt(prompt, fill_token=const.PROMPT_NOISE_FILLER_TOKEN)
                if  _valid_string(prompt):
                    prompts_map[lang][nls_label] = prompt

    for lang in supported_languages:
        missing_prompts_map[lang].append(list(nls_labels - set(prompts_map[lang].keys())))

    return prompts_map, missing_prompts_map


def setup_prompts(args: argparse.Namespace) -> None:
    
    dataset: str = args.file
    overwrite: bool = args.overwrite or True
    dest: str = os.path.join(args.dest,"prompts.yaml") if args.dest else const.PROMPTS_CONFIG_PATH

    if not os.path.exists(dataset):
        raise RuntimeError(
            f"""
            Invalid input, file does not exist {dataset}.
            """.strip()
        )

    if not (dataset.endswith(".yaml") or dataset.endswith(".csv")):
        raise RuntimeError(
            f"""
            Invalid input, pass either a .csv or .yaml file.
            """.strip()
        )

    if not os.path.exists(dest):
        os.mkdir(dest)

    if os.path.exists(os.path.join(dest,'prompts.yaml')) and not overwrite:
        raise RuntimeError(
            f"""
            File already exists {os.path.join(dest,'prompts.yaml')}.
            Use --overwrite=True
            """.strip()
        )
        
    data_frame = _nls_to_df(dataset) if dataset.endswith(".yaml") else pd.read_csv(dataset)

    if (const.STATE not in data_frame.columns and const.NLS_LABEL in data_frame.columns):
            logger.debug(f"State column missing, deriving from NLS-Keys")
            data_frame[const.STATE] = data_frame[const.NLS_LABEL].apply(lambda x: _nls_to_state(x))

    validate(data_frame)
    prompts_map, missing_prompts_map = get_prompts_map(data_frame)

    with open(dest, 'w') as file:
        yaml.safe_dump(prompts_map, file, allow_unicode=True)
    with open(dest.replace("prompts.yaml","missing_prompts.yaml"), 'w') as file:
        yaml.safe_dump(missing_prompts_map, file, allow_unicode=True)        
