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
import json

import pandas as pd
from dialogy.utils import create_timestamps_path
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate
from tqdm import tqdm

from slu import constants as const
from slu.dev.version import check_version_save_config
from slu.src.controller.prediction import get_predictions
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


def make_errors_report(
    config: Config, version: str, test_df: pd.DataFrame, predictions_df: pd.DataFrame
):
    logger.info(f"{test_df.head()}")
    test_df_ = test_df.copy()
    merged_df = pd.merge(
        test_df_, predictions_df, on="data_id", suffixes=("_test", "_pred")
    )
    errors_df = merged_df[
        merged_df[f"{const.INTENT}_test"] != merged_df[f"{const.INTENT}_pred"]
    ].copy()
    errors_df.to_csv(
        create_timestamps_path(
            config.get_metrics_dir(const.CLASSIFICATION, version=version),
            "error_report.csv",
        )
    )


def make_confusion_matrix(
    config: Config, version: str, test_df: pd.DataFrame, predictions_df: pd.DataFrame
):
    true_labels = zoom_out_labels(test_df[const.INTENT])
    pred_labels = zoom_out_labels(predictions_df[const.INTENT])
    labels = sorted(set(true_labels + pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    logger.info(f"Confusion matrix.\n{cm_df}")
    cm_df.to_csv(
        create_timestamps_path(
            config.get_metrics_dir(const.CLASSIFICATION, version=version),
            "confusion_matrix.csv",
        )
    )


def test_classifier(args: argparse.Namespace):
    """
    Evaluate the workflow with all the embedded plugins.

    Plugins can be evaluated individually for fine-tuning but since there are interactions
    between them, we need to evaluate them all together. This helps in cases where these interactions
    are a cause of a model's poor performance.

    This method doesn't mutate the given test dataset, instead we produce results with the same `id_`
    so that they can be joined and studied together.
    """
    version = args.version
    dataset = args.file
    lang = args.lang
    project_config_map = YAMLLocalConfig().generate()
    config: Config = list(project_config_map.values()).pop()
    check_version_save_config(config, version)

    predict_api = get_predictions(const.TEST, config=config, debug=False)
    dataset = dataset or config.get_dataset(const.CLASSIFICATION, f"{const.TEST}.csv")
    test_df = pd.read_csv(dataset)

    logger.info("Running predictions")
    predictions = []
    logger.disable("slu")

    for i, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        output = predict_api(
            **{
                const.ALTERNATIVES: json.loads(row[const.ALTERNATIVES]),
                const.CONTEXT: {},
                const.LANG: lang,
                "ignore_test_case": True,
            }
        )
        intents = output.get(const.INTENTS, [])
        predictions.append(
            {
                "data_id": row["data_id"],
                const.INTENT: intents[0][const.NAME] if intents else "_no_preds_",
                const.SCORE: intents[0][const.SCORE] if intents else 0,
            }
        )

    logger.enable("slu")
    predictions_df = pd.DataFrame(predictions)
    make_errors_report(config, version, test_df, predictions_df)
    make_classification_report(config, version, test_df, predictions_df)
    make_confusion_matrix(config, version, test_df, predictions_df)
