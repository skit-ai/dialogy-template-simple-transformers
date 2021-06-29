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
import shutil

import semver
from docopt import docopt
from sklearn import preprocessing  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from slu import constants as const
from slu.utils.config import Config
from slu.utils.logger import log


def train_intent_classifier(
    config: Config, eval_split=const.N_EVAL_SPLIT, file_format=const.CSV
):
    log.info("Preparing dataset.")
    train_df = config.get_dataset(
        const.CLASSIFICATION, const.TRAIN, file_format=file_format
    )

    label_encoder = preprocessing.LabelEncoder()
    encoder = label_encoder.fit(train_df[const.LABELS])
    train_df[const.LABELS] = encoder.transform(train_df[const.LABELS])
    config.set_labels(const.CLASSIFICATION, const.TRAIN, label_encoder)

    train_df, eval_df = train_test_split(
        train_df,
        test_size=eval_split,
        random_state=0,
        stratify=train_df[[const.LABELS]],
    )

    model = config.get_model(const.CLASSIFICATION, const.TRAIN)
    model_dir = config.get_model_dir(const.CLASSIFICATION, const.TRAIN)

    log.info("Training started.")
    model.train_model(
        train_df,
        output_dir=model_dir,
        eval_df=eval_df,
        acc=accuracy_score,
    )

    log.info("Saving artifacts.")
    config.save()
    config.remove_checkpoints(const.CLASSIFICATION)

    log.debug("Finished!")


def train_ner_model(config: Config, file_format=const.CSV):
    log.info("Preparing dataset.")
    train_df = config.get_dataset(const.NER, const.TRAIN, file_format=file_format)
    labels = sorted(train_df[const.LABELS].unique().tolist())
    config.set_labels(const.NER, const.TRAIN, labels)
    model = config.get_model(const.NER, const.TRAIN)

    log.info("Training started.")
    model.train_model(train_df)

    log.info("Saving artifacts.")
    config.save()
    config.remove_checkpoints(const.NER, const.TRAIN)

    log.debug("Finished!")
