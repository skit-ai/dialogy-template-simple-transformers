"""
Generate config for prompts using nls-keys downloaded from studio
Usage:
    prompt_setup.py
Options:
    -h --help   Show this screen.
    --version   Show version.
"""

import argparse
import os
import re
import string

import pandas as pd
import yaml
from tqdm import tqdm

from slu import constants as const
from slu.utils import logger
from slu.utils.config import Config, YAMLLocalConfig
from slu.utils.validations import valid_string


def preprocess_prompt(
    prompt: str,
    remove_var: bool = True,
    fill_token: str = const.PROMPT_NOISE_FILLER_TOKEN,
) -> str:

    """
    REGEX1: Detect special characters
    REGEX2: Detect noisy digits, alphanumberic characters
    REGEX3: Detect consequtive blank spaces
    REGEX4: Detect noisy addition (+) sequences. 
    REGEX5: Detect independent underscore (_) characters. 
    REGEX6: Detect variables defined inside prompts. Eg: {{.variable}}
    """

    REGEX1 = re.compile("\[.*?\]")
    REGEX2 = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    REGEX3 = re.compile("/\W/g")
    REGEX4 = re.compile("\+\s")
    REGEX5 = re.compile("\_")
    REGEX6 = re.compile(r"{{(.*?)}}")

    prompt = prompt.lower()
    prompt = re.sub(REGEX1, "", prompt)
    prompt = re.sub(REGEX2, "", prompt)
    prompt = re.sub(REGEX3, "", prompt)
    prompt = re.sub(REGEX4, " ", prompt)
    prompt = re.sub(REGEX5, " ", prompt)
    prompt = prompt.strip()
    if remove_var:
        prompt = re.sub(REGEX6, fill_token, prompt)
    prompt = prompt.translate(
        str.maketrans("", "", string.punctuation.replace("<", "").replace(">", ""))
    )

    return prompt

def nls_to_state(string: str, delimiter: str = "_") -> str:
    """
    Convert an NLS-Label to its corresponding State.
    NLS-Labels are nothing but extentions of State names.
    Eg: state "A" will have NLS-Label names A_1, A_2, and so on.
    This code simply removes the suffix _1, _2, etc from a NLS-Key to get the original State name.
    """
    if not valid_string(string):
        return None
    string = string.split(delimiter)
    if len(string) > 1:
        string = str(delimiter).join(_ for _ in string[: len(string) - 1])
    if isinstance(string, list):
        string = string[0]

    return string


def nls_to_df(dataset: str, config: Config) -> pd.DataFrame:
    """
    Convert an unstructured Flow-Creator file (.yaml) to Dataframe.
    """
    
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
    
    for lang in config.get_supported_languages():
        if const.PLATFORM_LEVEL_NOISE.get(lang):
            for _ in const.PLATFORM_LEVEL_NOISE[lang]:
                if _ in nls_labels:
                    nls_labels[lang] = nls_labels[_]
                    del nls_labels[_]
                
        if lang not in nls_labels:
            raise Exception(
                f"No prompts found for {lang}, please check your input file."
            )

        else:
            for _ in nls_labels[lang].keys():
                nls_keys.add(_)

    logger.debug(f"Total unique nls-keys: {len(nls_keys)}")

    nls_df = pd.DataFrame(
        columns=[const.NLS_LABEL] + list(config.get_supported_languages())
    )
    nls_df[const.NLS_LABEL] = pd.Series(list(nls_keys))

    for i in tqdm(range(nls_df.shape[0]), desc="Fetching prompts"):
        nls_label = nls_df.iloc[i][const.NLS_LABEL]
        for lang in config.get_supported_languages():
            if nls_label not in nls_labels[lang]:
                logger.debug(f"nls-key  {nls_label} not found for lang {lang}")
            elif not nls_labels[lang][nls_label]:
                logger.debug(f"Prompt not found for lang {lang}, nls-key {nls_label}")
            else:
                if (
                    isinstance(nls_labels[lang][nls_label], str)
                    and len(nls_labels[lang][nls_label]) > 0
                ):
                    nls_df.at[i, lang] = nls_labels[lang][
                        nls_label
                    ]
                if (
                    isinstance(
                        nls_labels[lang][nls_label], list
                    )
                    and len(nls_labels[lang][nls_label]) == 1
                ):
                    nls_df.at[i, lang] = nls_labels[lang][
                        nls_label
                    ][0]

    return nls_df


def validate(df: pd.DataFrame) -> None:

    if not const.NLS_LABEL in df.columns:
        raise Exception(f"Mandatory column missing {const.NLS_LABEL}")
    if not const.STATE in df.columns:
        raise Exception(f"Mandatory column missing {const.STATE}")

    df = df[(df[const.NLS_LABEL].notna()) & (df[const.STATE].notna())]
    invalid_rows = df[(df[const.NLS_LABEL].isna()) | (df[const.STATE].isna())]
    logger.debug(f"Num invalid rows: {invalid_rows.shape[0]}")


def get_prompts_map(df: pd.DataFrame) -> tuple[dict,dict]:
    """
    Extract prompts, missing prompts from a Dataframe.
    missing_prompts: Sometimes an nls_label maybe defined for English, but not present for Hindi. missing_prompts gives a list of such nls_labels. 
    """
    
    prompts_map: dict = dict()
    missing_prompts_map: dict = dict()
    supported_languages: list = [
        col
        for col in df.columns
        if (valid_string(col) and col not in [const.NLS_LABEL, const.STATE])
    ]
    nls_labels: set = set()

    if not supported_languages:
        raise Exception("No languages found in the dataset")

    logger.debug(f"Found languages: {supported_languages}")
    for lang in supported_languages:
        prompts_map[lang] = {}
        missing_prompts_map[lang] = []

        for i in tqdm(range(df.shape[0]), desc=f"Fetching prompts for {lang}"):
            nls_label = df.iloc[i][const.NLS_LABEL]
            nls_labels.add(nls_label)
            prompt = df.iloc[i][lang]

            if valid_string(prompt):
                prompt = preprocess_prompt(
                    prompt
                )
                if valid_string(prompt):
                    prompts_map[lang][nls_label] = prompt

    for lang in supported_languages:
        missing_prompts_map[lang].append(
            list(nls_labels - set(prompts_map[lang].keys()))
        )

    return prompts_map, missing_prompts_map


def setup_prompts(args: argparse.Namespace) -> None:
    """
    Create prompts.yaml from a raw file (Can be a .csv downloaded from Studio, or .yaml fetched from Flow-Creator).
    """

    dataset: str = args.file
    overwrite: bool = args.overwrite
    dest_p: str = (
        os.path.join(args.dest, "prompts.yaml")
        if args.dest
        else const.PROMPTS_CONFIG_PATH
    )
    dest_mp: str = (
        os.path.join(args.dest, "missing_prompts.yaml")
        if args.dest
        else const.MISSING_PROMPTS_PATH
    )
        
    project_config_map = YAMLLocalConfig().generate()
    config: Config = list(project_config_map.values()).pop()
    
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
    
    if os.path.exists(dest_p) and not overwrite:
        raise RuntimeError(
            f"""
            File already exists {dest_p}.
            Use --overwrite=True
            """.strip()
        )

    data_frame = (
        nls_to_df(dataset, config)
        if dataset.endswith(".yaml")
        else pd.read_csv(dataset)
    )

    if const.STATE not in data_frame.columns and const.NLS_LABEL in data_frame.columns:
        logger.debug(f"State column missing, deriving from NLS-Keys")
        data_frame[const.STATE] = data_frame[const.NLS_LABEL].apply(
            lambda x: nls_to_state(x)
        )

    validate(data_frame)
    prompts_map, missing_prompts_map = get_prompts_map(data_frame)

    with open(dest_p, "w") as file:
        yaml.safe_dump(prompts_map, file, allow_unicode=True)
    with open(dest_mp, "w") as file:
        yaml.safe_dump(missing_prompts_map, file, allow_unicode=True)
