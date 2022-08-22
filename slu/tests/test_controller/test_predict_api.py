import json

import pandas as pd
from tqdm import tqdm

from slu.src.controller.prediction import get_predictions
from slu.utils.config import Config, YAMLLocalConfig
from slu import constants as const
from slu.dev.test import make_classification_report


CONFIG_MAP = YAMLLocalConfig().generate()
CONFIG: Config = list(CONFIG_MAP.values()).pop()
PREDICT_API = get_predictions(const.PRODUCTION, config=CONFIG)


def test_classifier():
    """
    Evaluate the workflow with all the embedded plugins.

    Plugins can be evaluated individually for fine-tuning but since there are interactions
    between them, we need to evaluate them all together. This helps in cases where these interactions
    are a cause of a model's poor performance.

    This method doesn't mutate the given test dataset, instead we produce results with the same `id_`
    so that they can be joined and studied together.
    """
    project_config_map = YAMLLocalConfig().generate()
    config: Config = list(project_config_map.values()).pop()

    predict_api = get_predictions(const.PRODUCTION, config=config, debug=False)
    dataset = config.get_dataset(const.CLASSIFICATION, f"{const.TRAIN}.csv")
    test_df = pd.read_csv(dataset).sample(n=100)

    predictions = []
    config.tasks.classification.threshold = 0

    for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        output = predict_api(
            **{
                const.ALTERNATIVES: json.loads(row[const.ALTERNATIVES]),
                const.CONTEXT: {},
                const.LANG: "en",
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

    predictions_df = pd.DataFrame(predictions)
    report = make_classification_report(test_df, predictions_df)
    assert report["f1-score"]["weighted avg"] >= 0.9
