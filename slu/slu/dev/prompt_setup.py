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
from typing import Dict, Tuple
import pandas as pd
import yaml
from tqdm import tqdm
from loguru import logger

from slu import constants as const
from slu.utils import logger
from slu.utils.config import Config, YAMLLocalConfig, load_prompt_config
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

def nls_to_state(string: str, delimiter: str = "_") -> Optional[str]:
    """
    Convert an NLS-Label to its corresponding State.
    - NLS-Labels are nothing but extentions of State names.
    - Eg: state "A" will have NLS-Label names A_1, A_2, and so on.
    - This code simply removes the suffix _1, _2, etc from a NLS-Key to get the original State name.
    - Returns an empty string in case of an invalid NLS label. 
    """
    if not valid_string(string):
        return None
    
    string = string.split(delimiter)
    if len(string) > 1:
        string = str(delimiter).join(ch for ch in string[: len(string) - 1])
    if isinstance(string, list):
        string = string[0]

    return string

def state_to_nls(current_state: str, nls_labels: set)-> str:
    """
    Get the best nls_label match, given current-state and list of available nls labels.
    Returns an empty string if match not found.
    """
    for nls_label in list(nls_labels):
        if nls_label == current_state:
            return nls_label
        else:
         state = nls_to_state(nls_label)
         if state == current_state:
             return nls_label
    return ''

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
            for term in const.PLATFORM_LEVEL_NOISE[lang]:
                if term in nls_labels:
                    nls_labels[lang] = nls_labels[term]
                    del nls_labels[term]
                
        if lang not in nls_labels:
            raise Exception(
                f"No prompts found for {lang}, please check your input file."
            )

        else:
            for label in nls_labels[lang].keys():
                nls_keys.add(label)

    logger.debug(f"Total unique nls-keys: {len(nls_keys)}")

    nls_df = pd.DataFrame(
        columns=[const.NLS_LABEL] + list(config.get_supported_languages())
    )
    nls_df[const.NLS_LABEL] = pd.Series(list(nls_keys))

    for i in tqdm(range(nls_df.shape[0]), desc="Fetching prompts"):
        """
        By iterating over the dataframe, we are trying to fetch prompts in each language for the given nls_label.
        A nls_label may or may NOT have prompts defined for all suppoorted languages.
        """
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


def get_prompts_map(df: pd.DataFrame) -> Tuple[Dict[str, str],Dict[str,str]]:
    """
    Extract prompts, missing prompts from a Dataframe.
    missing_prompts: Sometimes an nls_label maybe defined for English, but not present for Hindi. missing_prompts gives a list of such nls_labels. 
    """
    tqdm.pandas()
    prompts_map: dict = dict()
    missing_prompts_map: dict = dict()
    supported_languages: list = [
        col
        for col in df.columns
        if (valid_string(col) and col not in [const.NLS_LABEL, const.STATE])
    ]
    nls_labels: set = set(df[const.NLS_LABEL].unique())

    if not supported_languages:
        raise Exception("No languages found in the dataset")

    logger.debug(f"Found languages: {supported_languages}")
    for lang in supported_languages:
        """
        Preprocess raw prompts.
        Map nls labels -> prompts for each supported language (eg: en, hi) and store in prompts_map.
        """
        prompts_map[lang] = {}
        missing_prompts_map[lang] = []
        
        df[lang] = df[lang].progress_apply(lambda prompt: preprocess_prompt(prompt) if valid_string(prompt) else 'nan')
        prompts_map[lang] = {nls_label: prompt for nls_label, prompt in zip(df[const.NLS_LABEL], df[lang])}

        missing_prompts_map[lang].append(
            list(nls_labels - set(prompts_map[lang].keys()))
        )

    return prompts_map, missing_prompts_map


def setup_prompts(args: argparse.Namespace) -> None:
    """
    Create prompts.yaml from a raw file (Can be a .csv downloaded from Studio, or .yaml fetched from Flow-Creator).
    The final output, i.e. prompts.yaml will have the following structure:
    {
        en:
            nls_label1: prompt1
            nls_label2: prompt2
        
        hi:
            nls_label1: prompt3
            nls_label2: prompt4  
    }    
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


def fill_nls_col(args: argparse.Namespace)-> None:
    """
    This is a temporary workaround to handle datasets with missing nls_label info.
    """
    input_file: str = args.input_file
    output_file:str = args.output_file if args.output_file else args.input_file
    overwrite: bool = args.overwrite
    nls_labels = set()
    tqdm.pandas()

    if not (input_file.endswith(".csv")):
        raise RuntimeError(
            f"""
            Invalid input, pass .csv file.
            """.strip()
        )
    if not (output_file.endswith(".csv")):
        raise RuntimeError(
            f"""
            Invalid output, must be a .csv file.
            """.strip()
        )
    if os.path.exists(output_file) and not overwrite:
        raise RuntimeError(
            f"""
            Specificy --output_file or use --overwrite=True
            """.strip()
        )
    
    data_frame = pd.read_csv(input_file)
    if not const.STATE in data_frame.columns:
        raise RuntimeError(
            f"""
            {const.STATE} column missing in {input_file}. 
            Make sure it is populated
            """.strip()
        )
        
    prompts_map: dict = load_prompt_config()
    for key in prompts_map:
        for nls_label in prompts_map[key]:
            nls_labels.add(nls_label)
    data_frame[const.NLS_LABEL] = data_frame[const.STATE].progress_apply(lambda current_state: state_to_nls(current_state,nls_labels))
    data_frame.to_csv(output_file, index=False)
    
    logger.debug(f"Saved to {output_file}.")