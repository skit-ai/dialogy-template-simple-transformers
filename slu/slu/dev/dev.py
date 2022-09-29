import os
import argparse
import json

import numpy as np
import pandas as pd
from dialogy import plugins
from dialogy.utils import (
    create_timestamps_path,
    save_to_json,
    fit_ts_parameter,
    save_reliability_graph,
)
import dialogy.constants as dialogy_const

from tqdm import tqdm
from slu import constants as const
from slu.dev.version import check_version_save_config
from slu.src.controller.prediction import get_predictions
from slu.utils import logger
from slu.utils.config import Config, load_gen_config


def get_onehot(targets: np.ndarray, nb_classes: int):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape).append(nb_classes))


def form_ordered_onehot_n_logits(
    true_intents: pd.Series, all_pred_intents: np.ndarray, all_logits: np.ndarray
):
    all_classes = true_intents.unique()
    class_to_num_map = {
        class_: all_classes.tolist().index(class_) for class_ in all_classes
    }
    labels_oneh = []
    all_ordered_logits = []
    encoded_labels_list = np.array(
        [class_to_num_map[intent_] for intent_ in true_intents]
    )
    labels_oneh = get_onehot(targets=encoded_labels_list, nb_classes=len(all_classes))
    for pred_intents, logits in zip(all_pred_intents, all_logits):
        ordered_logits = np.array(
            [
                logit
                for _, logit in sorted(
                    zip(pred_intents, logits),
                    key=lambda pair: class_to_num_map[pair[0]],
                )
            ]
        )
        all_ordered_logits.append(ordered_logits)
    return np.array(all_ordered_logits), labels_oneh.flatten(), encoded_labels_list


def save_calibration_stuff(
    predictions_df: pd.DataFrame, to_calibrate: bool, dir_path: str, prefix: str = ""
):
    exclude = ["_error_", "_no_preds_"]
    predictions_df = predictions_df[
        predictions_df["prediction"].apply(
            lambda pred: not any(p_[const.NAME] in exclude for p_ in json.loads(pred))
        )
    ]
    all_true_intents = predictions_df[const.TAG].copy()
    all_pred_intents, all_logits = zip(
        *predictions_df["prediction"].apply(
            lambda pred: list(
                zip(
                    *(
                        (pred_candidate[const.NAME], pred_candidate[const.SCORE])
                        for pred_candidate in json.loads(pred)
                    )
                )
            )
        )
    )
    (
        all_ordered_logits,
        all_ordered_labels_oneh,
        all_labels_list,
    ) = form_ordered_onehot_n_logits(all_true_intents, all_pred_intents, all_logits)
    if to_calibrate:
        ts_parameter = fit_ts_parameter(all_ordered_logits, all_labels_list)
        save_to_json(
            {dialogy_const.TS_PARAMETER: ts_parameter},
            dir_path,
            file_name=dialogy_const.CALIBRATION_CONFIG_FILE,
        )
    else:
        save_reliability_graph(
            all_ordered_logits.flatten(), all_ordered_labels_oneh, dir_path, prefix
        )


def get_test_outputs(
    config: Config,
    test_df: pd.DataFrame,
    lang: str,
):
    calibration_predictions = []
    PREDICT_API = get_predictions(
        purpose=const.TEST,
        final_plugin=plugins.XLMRMultiClass,
        config=config,
        debug=False,
    )
    logger.disable("slu")
    for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        output = PREDICT_API(
            **{
                const.ALTERNATIVES: json.loads(row[const.ALTERNATIVES]),
                const.CONTEXT: {},
                const.LANG: lang,
                "ignore_test_case": True,
            }
        )
        intents = output.get(const.INTENTS, [])
        calibration_predictions.append(
            {
                "data_id": row["data_id"] or row["conversation_uuid"],
                "state": row["state"],
                const.ALTERNATIVES: row[const.ALTERNATIVES],
                const.TAG: row[const.TAG],
                "prediction": json.dumps(
                    [
                        {
                            const.NAME: intent[const.NAME] if intent else "_no_preds_",
                            const.SCORE: intent[const.SCORE] if intent else 0,
                            const.SLOTS: intent[const.SLOTS] if intent else [],
                        }
                        for intent in intents
                    ],
                    ensure_ascii=False,
                )
                if intents
                else json.dumps(
                    [{const.NAME: "_no_preds_", const.SCORE: 0, const.SLOTS: []}],
                    ensure_ascii=False,
                ),
            }
        )
    logger.enable("slu")
    return calibration_predictions


def temperature_scaling_processor(
    config: Config,
    test_df: pd.DataFrame,
    lang: str,
    to_calibrate: bool,
    dir_path: str,
    prefix: str = "",
):
    calibration_predictions = get_test_outputs(config, test_df, lang)
    save_calibration_stuff(
        predictions_df=pd.DataFrame(calibration_predictions),
        to_calibrate=to_calibrate,
        dir_path=dir_path,
        prefix=prefix,
    )


def dev_workflow(args: argparse.Namespace):
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
    calibrate = args.calibrate
    config: Config = load_gen_config()
    check_version_save_config(config, version)
    dataset = dataset or config.get_dataset(const.CLASSIFICATION, f"{const.TEST}.csv")

    test_df = pd.read_csv(dataset)
    metrics_dir_path = create_timestamps_path(
        config.get_metrics_dir(const.CLASSIFICATION, version=version),
        "",
    )
    models_dir_path = config.get_model_dir(const.CLASSIFICATION, version=version)
    logger.info("Running predictions")

    if calibrate:
        # save current reliability graph
        logger.info("Generating and saving current reliability graph")
        temperature_scaling_processor(
            config=config,
            test_df=test_df,
            lang=lang,
            to_calibrate=False,
            dir_path=metrics_dir_path,
            prefix="old",
        )

        # calibrate
        logger.info("Calibrating your model")
        config.tasks.classification.model_args[const.TEST][
            dialogy_const.MODEL_CALIBRATION
        ] = True
        temperature_scaling_processor(
            config=config,
            test_df=test_df,
            lang=lang,
            to_calibrate=True,
            dir_path=models_dir_path,
        )

        # save calibrated reliability graph
        logger.info("Generating and saving calibrated reliability graph")
        config.tasks.classification.model_args[const.TEST][
            dialogy_const.MODEL_CALIBRATION
        ] = False
        temperature_scaling_processor(
            config=config,
            test_df=test_df,
            lang=lang,
            to_calibrate=False,
            dir_path=metrics_dir_path,
            prefix="new",
        )
