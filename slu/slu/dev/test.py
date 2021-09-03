"""
Testing routine.

Usage:
  test.py <version>
  test.py (classification|ner) <version>
  test.py (-h | --help)
  test.py --version

Options:
    <version>   The version of the dataset to use, the model produced will also be in the same dir.
    -h --help   Show this screen.
    --version   Show version.
"""
import argparse

import pandas as pd
import semver
from dialogy.utils import create_timestamps_path
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate

from slu import constants as const
from slu.src.controller.prediction import get_workflow
from slu.utils import logger
from slu.utils.config import Config, YAMLLocalConfig


def zoom_out_labels(labels):
    """

    :param labels: [description]
    :type labels: [type]
    """
    labels_ = []
    for label in labels:
        if label == "_oos_":
            labels_.append("out-of-scope")
        elif label.startswith("_") and label.endswith("_"):
            labels_.append(label)
        else:
            labels_.append("in-scope")
    return labels_


def make_classification_report(
    config: Config, version: str, test_df: pd.DataFrame, predictions_df: pd.DataFrame
):
    result_dict = classification_report(
        test_df[const.INTENT],
        predictions_df[const.INTENT],
        zero_division=0,
        output_dict=True,
    )
    result_df = pd.DataFrame(result_dict).T
    logger.info("saving report.")
    table = tabulate(result_df, headers="keys", tablefmt="github")
    logger.info(f"classification report:\n{table}")

    result_df.to_csv(
        create_timestamps_path(
            config.get_metrics_dir(const.CLASSIFICATION, version=version),
            "classification_report.csv",
        )
    )


def make_confusion_matrix(
    config: Config, version: str, test_df: pd.DataFrame, predictions_df: pd.DataFrame
):
    true_labels = zoom_out_labels(test_df[const.INTENT])
    pred_labels = zoom_out_labels(predictions_df[const.INTENT])
    labels = sorted(true_labels + pred_labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    logger.info(f"Confusion matrix.\n{cm}")
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(
        create_timestamps_path(
            config.get_metrics_dir(const.CLASSIFICATION, version=version),
            "confusion_matrix.csv",
        )
    )


def test_classifier(args: argparse.Namespace):
    version = args.version
    dataset = args.file
    project_config_map = YAMLLocalConfig().generate()
    config: Config = list(project_config_map.values()).pop()
    if version:
        semver.VersionInfo.parse(version)
        config.version = version
        config.save()

    workflow = get_workflow(const.TEST, lang=args.lang, project=args.project)

    dataset = dataset or config.get_dataset(const.CLASSIFICATION, f"{const.TEST}.csv")
    test_df = pd.read_csv(dataset)

    logger.info("Running predictions")
    predictions_df = workflow.prediction_labels(test_df, id_="data_id")
    make_classification_report(config, version, test_df, predictions_df)
    make_confusion_matrix(config, version, test_df, predictions_df)
