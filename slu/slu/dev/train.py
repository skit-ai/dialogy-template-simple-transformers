"""
Routine for Classifier and NER training. Provide a version and a model will be trained on a dataset
of the same version.

This script expects data/<version> to be a directory where models, metrics and dataset are present.

Usage:
  train.py <version>
  train.py (classification|ner) <version>
  train.py (-h | --help)
  train.py --version

Options:
    <version>     The version of the dataset to use, the model produced will also be in the same dir.
    -h --help     Show this screen.
    --version     Show version.
"""
import argparse
import json
import os

import pandas as pd
import semver
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from slu import constants as const
from slu.dev.version import check_version_save_config
from slu.src.controller.prediction import get_workflow
from slu.utils import logger
from slu.utils.config import Config, YAMLLocalConfig


def make_label_column_uniform(data_frame: pd.DataFrame) -> None:
    if const.INTENT in data_frame.columns:
        column = const.INTENT
    elif const.INTENTS in data_frame.columns:
        column = const.INTENTS
    elif const.LABELS in data_frame.columns:
        column = const.LABELS
    elif const.TAG in data_frame.columns:
        column = const.TAG
    else:
        raise ValueError(
            f"Expected one of {const.INTENT}, {const.LABELS}, {const.TAG} to be present in the dataset."
        )
    data_frame.rename(columns={column: const.INTENT}, inplace=True)


def make_data_column_uniform(data_frame: pd.DataFrame) -> None:
    if const.ALTERNATIVES in data_frame.columns:
        column = const.ALTERNATIVES
    elif const.DATA in data_frame.columns:
        column = const.DATA
    else:
        raise ValueError(
            f"Expected one of {const.ALTERNATIVES}, {const.DATA} to be present in the dataset."
        )
    data_frame.rename(columns={column: const.ALTERNATIVES}, inplace=True)

    for i, row in tqdm(
        data_frame.iterrows(), total=len(data_frame), desc="Fixing data structure"
    ):
        if isinstance(row[const.ALTERNATIVES], str):
            data = json.loads(row[const.ALTERNATIVES])
            if const.ALTERNATIVES in data:
                data_frame.loc[i, const.ALTERNATIVES] = json.dumps(
                    data[const.ALTERNATIVES]
                )


def create_data_splits(args: argparse.Namespace) -> None:
    """
    Create a data split for the given version.
    :param args: The arguments passed to the script.
    """
    version = args.version
    project_config_map = YAMLLocalConfig().generate()
    config: Config = list(project_config_map.values()).pop()
    check_version_save_config(config, version)

    dataset_file = args.file
    train_size = args.train_size
    test_size = args.test_size
    stratify = args.stratify
    dest = args.dest or config.get_dataset_dir(const.CLASSIFICATION)

    if os.listdir(dest):
        ver_ = semver.VersionInfo.parse(config.version)
        ver_.bump_patch()
        raise RuntimeError(
            f"""
Data already exists in {dest} You should create a new version using:

```shell
slu dir-setup --version {str(ver_.bump_patch())}
```
""".strip()
        )

    if not os.path.isdir(dest):
        raise ValueError(
            f"Destination directory {dest} does not exist or is not a directory."
        )

    data_frame = pd.read_csv(dataset_file)
    logger.debug(f"Data frame: {data_frame.shape}")
    skip_list = config.get_skip_list(const.CLASSIFICATION)

    make_label_column_uniform(data_frame)
    make_data_column_uniform(data_frame)

    skip_filter = data_frame[const.INTENT].isin(skip_list)
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
        labels = data_frame[stratify].unique()
    else:
        labels = None

    train, test = train_test_split(
        train_available_samples,
        train_size=train_size,
        test_size=test_size,
        stratify=labels,
    )
    test = pd.concat([train_skip_samples, test], sort=False)
    train.to_csv(os.path.join(dest, f"{const.TRAIN}.csv"), index=False)
    test.to_csv(os.path.join(dest, f"{const.TEST}.csv"), index=False)


def merge_datasets(args: argparse.Namespace) -> None:
    """
    Merge the datasets.
    """
    data_files = args.files
    file_name = args.out

    data_frames = pd.concat([pd.read_csv(data_file) for data_file in data_files])
    data_frames.to_csv(file_name, index=False)


def train_intent_classifier(args: argparse.Namespace) -> None:
    version = args.version
    dataset = args.file
    project_config_map = YAMLLocalConfig().generate()
    config: Config = list(project_config_map.values()).pop()
    check_version_save_config(config, version)

    model_dir = config.get_model_dir(const.CLASSIFICATION)
    if os.listdir(model_dir):
        ver_ = semver.VersionInfo.parse(config.version)
        ver_.bump_patch()
        raise RuntimeError(
            f"""
Model already exists in {model_dir}.
You should create a new version using:

```shell
slu dir-setup --version {str(ver_.bump_patch())}
```
""".strip()
        )

    workflow = get_workflow(const.TRAIN, lang=args.lang, project=args.project)

    logger.info("Preparing dataset.")
    dataset = dataset or config.get_dataset(const.CLASSIFICATION, f"{const.TRAIN}.csv")
    train_df = pd.read_csv(dataset)

    logger.info("Training started.")
    workflow.train(train_df)
    config.save()
    logger.debug("Finished!")
