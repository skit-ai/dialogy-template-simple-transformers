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

import pandas as pd

from slu import constants as const
from slu.src.controller.prediction import get_workflow
from slu.utils.config import Config, YAMLLocalConfig
from slu.utils import logger


def train_intent_classifier(args: argparse.Namespace) -> None:
    version = args.version
    dataset = args.file
    project_config_map = YAMLLocalConfig().generate()
    config: Config = list(project_config_map.values()).pop()

    workflow = get_workflow(const.TRAIN, lang=args.lang, project=args.project)

    logger.info("Preparing dataset.")
    dataset = dataset or config.get_dataset(const.CLASSIFICATION, f"{const.TRAIN}.csv")
    train_df = pd.read_csv(dataset)

    logger.info("Training started.")
    workflow.train(train_df)
    config.save()
    logger.debug("Finished!")
