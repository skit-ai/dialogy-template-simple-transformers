from slu.utils.preprocessing import make_label_column_uniform, make_data_column_uniform, make_reftime_column_uniform
"""
Routine for Classifier and NER training.
"""

import argparse
import functools
import json
import os
from datetime import datetime

import pandas as pd
import semver
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from slu import constants as const
from slu.src.controller.processors import SLUPipeline
from slu.utils import logger
from slu.utils.config import Config, YAMLLocalConfig
from slu.utils.preprocessing import make_label_column_uniform, make_data_column_uniform, make_reftime_column_uniform

def create_data_splits(args: argparse.Namespace) -> None:
    """
    Create a data split for the given dataset.
    :param args: The arguments passed to the script.
    """
    project_config_map = YAMLLocalConfig().generate()
    config: Config = list(project_config_map.values()).pop()

    dataset_file = args.file
    train_size = args.train_size
    test_size = args.test_size
    stratify = args.stratify
    dest = args.dest or config.get_dataset_dir(const.CLASSIFICATION)

    if os.listdir(dest):
        raise RuntimeError(
            f"""
Data already exists in {dest}
""".strip()
        )

    if not os.path.isdir(dest):
        raise ValueError(
            f"Destination directory {dest} does not exist or is not a directory."
        )

    data_frame = pd.read_csv(dataset_file)
    logger.debug(f"Data frame: {data_frame.shape}")
    data_frame = make_label_column_uniform(data_frame, const.ALIAS_TRAIN_PATH)
    skip_list = config.get_skip_list(const.CLASSIFICATION)
    skip_filter = data_frame[const.TAG].isin(skip_list)
    logger.info(
        f"Model will be trained for the following classes:\
        \n{data_frame[const.TAG].value_counts(dropna=False)}"
    )
    
    failed_transcripts = data_frame[const.ALTERNATIVES].isin(["[[]]", "[]"])
    non_empty_transcripts = data_frame[const.ALTERNATIVES].isna()
    invalid_samples = skip_filter | non_empty_transcripts | failed_transcripts
    train_skip_samples = data_frame[invalid_samples]
    train_available_samples = data_frame[~invalid_samples]

    logger.info(
        f"Dataset has {len(train_skip_samples)} samples unfit for training."
        f" Using this for tests and {len(train_available_samples)} for train-test split."
    )

    if stratify:
        train_available_samples = train_available_samples[
            train_available_samples[const.TAG].map(
                train_available_samples[const.TAG].value_counts() > 1
            )
        ]
        labels = train_available_samples[const.TAG]

    else:
        labels = None

    train, test = train_test_split(
        train_available_samples,
        train_size=train_size,
        test_size=test_size,
        stratify=labels,
    )
    train.to_csv(os.path.join(dest, f"{const.TRAIN}.csv"), index=False)
    test.to_csv(os.path.join(dest, f"{const.TEST}.csv"), index=False)
    train_skip_samples.to_csv(os.path.join(dest, f"{const.SKIPPED}.csv"), index=False)


def merge_datasets(args: argparse.Namespace) -> None:
    """
    Merge the datasets.
    """
    data_files = args.files
    file_name = args.out

    data_frames = pd.concat([pd.read_csv(data_file) for data_file in data_files])
    data_frames.to_csv(file_name, index=False)


def train_intent_classifier(args: argparse.Namespace) -> None:
    dataset = args.file
    epochs = args.epochs
    project_config_map = YAMLLocalConfig().generate()
    config: Config = list(project_config_map.values()).pop()

    model_dir = config.get_model_dir(const.CLASSIFICATION)
    if os.listdir(model_dir):
        raise RuntimeError(
            f"""
            Model already exists in {model_dir}.
            """.strip()
        )

    workflow = SLUPipeline(config).get_workflow(purpose=const.TRAIN, epochs=epochs)

    logger.info("Preparing dataset.")
    dataset = dataset or config.get_dataset(const.CLASSIFICATION, f"{const.TRAIN}.csv")
    data_frame = pd.read_csv(dataset)
    data_frame = make_label_column_uniform(data_frame, const.ALIAS_TRAIN_PATH)
    data_frame = make_data_column_uniform(data_frame)
    data_frame = make_reftime_column_uniform(data_frame)
    logger.info(
        f"Model will be trained for the following classes:\
        \n{data_frame[const.TAG].value_counts(dropna=False)}"
    )

    logger.info("Training started.")
    workflow.train(data_frame)
    config.save()
    logger.debug("Finished!")
