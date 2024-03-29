"""
Testing routine.
"""

import argparse
import json
import os
from typing import List, Optional

import pandas as pd
from dialogy.utils import create_timestamps_path
from dialogy.base.output import Output
from pandas.core.reshape.merge import merge
from sklearn.metrics import classification_report, confusion_matrix
from tabulate import tabulate
from tqdm import tqdm

from slu import constants as const
from slu.src.controller.prediction import get_predictions
from slu.utils import logger
from slu.utils.config import Config, YAMLLocalConfig

def zoom_out_labels(labels: List[str]):
    """

    :param labels: [description]
    :type labels: [type]
    """
    labels_ = []
    for label in labels:
        if label == const.INTENT_OOS:
            labels_.append("out-of-scope")
        elif label.startswith("_") and label.endswith("_"):
            labels_.append(label)
        else:
            labels_.append("in-scope")
    return labels_


def update_confidence_scores(
    config: Config, test_df: pd.DataFrame, predictions_df: pd.DataFrame
):
    """
    Update the confidence scores in the config as per test results.
    """
    merged_df = pd.merge(test_df, predictions_df, on="data_id")
    valid_inputs = merged_df[~merged_df.alternatives.isin(["[]", "[[]]"])]
    correct_items = valid_inputs[valid_inputs.tag == valid_inputs.intent_pred]
    incorrect_items = valid_inputs[valid_inputs.tag != valid_inputs.intent_pred]
    logger.info(f"{correct_items.score.describe()}")
    logger.info(f"{incorrect_items.score.describe()}")


def make_classification_report(
    test_df: pd.DataFrame, predictions_df: pd.DataFrame, dir_path: Optional[str] = None
) -> pd.DataFrame:

    result_dict = classification_report(
        test_df[const.TAG],
        predictions_df[const.INTENT_PRED],
        zero_division=0,
        output_dict=True,
    )
    result_df = pd.DataFrame(result_dict).T
    logger.info("saving report.")
    table = tabulate(result_df, headers="keys", tablefmt="github")
    logger.info(f"classification report:\n{table}")

    if dir_path:
        result_df.to_csv(os.path.join(dir_path, "classification_report.csv"))

    return result_df


def make_critical_intent_report(
    test_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    critical_intents: List[str],
    dir_path: str,
):
    merged_df = pd.merge(test_df, predictions_df, on="data_id")
    merged_df = merged_df[
        (merged_df.tag.isin(critical_intents))
        | (merged_df.intent_pred.isin(critical_intents))
    ]
    result_dict = classification_report(
        merged_df.tag,
        merged_df.intent_pred,
        labels=critical_intents,
        zero_division=0,
        output_dict=True,
    )
    result_df = pd.DataFrame(result_dict).T
    logger.info("saving report.")
    table = tabulate(result_df, headers="keys", tablefmt="github")
    logger.info(f"classification report:\n{table}")
    result_df.to_csv(
        os.path.join(dir_path, "critical_intent_classification_report.csv")
    )


def make_errors_report(
    test_df: pd.DataFrame, predictions_df: pd.DataFrame, dir_path: str
):
    logger.info(f"{test_df.head()}")
    test_df_ = test_df.copy()
    merged_df = pd.merge(test_df_, predictions_df, on="data_id")
    errors_df = merged_df[
        merged_df[f"{const.TAG}"] != merged_df[const.INTENT_PRED]
    ].copy()
    true_labels = errors_df[f"{const.TAG}"].tolist()
    pred_labels = errors_df[const.INTENT_PRED].tolist()
    if not true_labels and not pred_labels:
        return
    make_confusion_matrix(true_labels, pred_labels, dir_path, prefix="errors")
    errors_df.to_csv(os.path.join(dir_path, "error_report.csv"))


def make_confusion_matrix(
    true_labels: List[str], pred_labels: List[str], dir_path: str, prefix=""
):
    labels = sorted(set(true_labels + pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    logger.info(f"Confusion matrix.\n{cm_df}")
    cm_df.to_csv(os.path.join(dir_path, f"{prefix}_confusion_matrix.csv"))


def test_classifier(args: argparse.Namespace):
    """
    Evaluate the workflow with all the embedded plugins.

    Plugins can be evaluated individually for fine-tuning but since there are interactions
    between them, we need to evaluate them all together. This helps in cases where these interactions
    are a cause of a model's poor performance.

    This method doesn't mutate the given test dataset, instead we produce results with the same `id_`
    so that they can be joined and studied together.
    """
    dataset = args.file
    lang = args.lang
    project_config_map = YAMLLocalConfig().generate()
    config: Config = list(project_config_map.values()).pop()

    predict_api = get_predictions(const.TEST, config=config, debug=False)
    dataset = dataset or config.get_dataset(const.CLASSIFICATION, f"{const.TEST}.csv")
    test_df = pd.read_csv(dataset)
    test_df = test_df[~test_df[const.TAG].isin(config.tasks.classification.skip)]
    test_df = test_df[test_df[const.ALTERNATIVES] != "[]"]
    test_df = test_df.replace({const.TAG: config.tasks.classification.alias})

    logger.info("Running predictions")
    predictions = []
    logger.disable("slu")
    logger.disable("dialogy")

    config.tasks.classification.threshold = 0

    for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        output = Output().dict()
        try:
            output = predict_api(
                **{
                    const.ALTERNATIVES: json.loads(row[const.ALTERNATIVES]),
                    const.CONTEXT: {
                        const.CURRENT_STATE: row.get(const.STATE),
                        const.NLS_LABEL: row.get(const.NLS_LABEL),
                    },
                    const.LANG: lang,
                    "ignore_test_case": True,
                }
            )
        except Exception as e:
            logger.enable("slu")
            logger.error(f"skipping {row[const.ALTERNATIVES]} because {e}")
            logger.disable("slu")

        intents = output.get(const.INTENTS, [])
        predictions.append(
            {
                "data_id": row["data_id"],
                const.INTENT_PRED: intents[0][const.NAME] if intents else "_no_preds_",
                const.SCORE: intents[0][const.SCORE] if intents else 0,
            }
        )

    logger.enable("slu")
    logger.enable("dialogy")

    predictions_df = pd.DataFrame(predictions)
    dir_path = create_timestamps_path(
        config.get_metrics_dir(const.CLASSIFICATION),
        "",
    )
    update_confidence_scores(config, test_df, predictions_df)
    make_errors_report(test_df, predictions_df, dir_path=dir_path)
    make_classification_report(test_df, predictions_df, dir_path=dir_path)
    make_critical_intent_report(
        test_df, predictions_df, config.critical_intents, dir_path=dir_path
    )

    true_labels = test_df[const.TAG].tolist()
    pred_labels = predictions_df[const.INTENT_PRED].tolist()
    zoomed_true_label = zoom_out_labels(true_labels)
    zoomed_predicted_label = zoom_out_labels(pred_labels)
    make_confusion_matrix(
        zoomed_true_label, zoomed_predicted_label, dir_path=dir_path, prefix="zoomed"
    )
    make_confusion_matrix(true_labels, pred_labels, dir_path=dir_path, prefix="full")
