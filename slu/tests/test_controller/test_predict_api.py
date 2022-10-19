import pytest
import os
import pandas as pd
from tqdm import tqdm
import json
import yaml

from slu.src.controller.prediction import get_predictions
from slu.utils.config import Config, YAMLLocalConfig
from slu import constants as const
from slu.dev.test import make_classification_report


@pytest.fixture(scope="session")
def slu_api():
    project_config_map = YAMLLocalConfig().generate()
    config: Config = list(project_config_map.values()).pop()

    predict_api = get_predictions(const.PRODUCTION, config=config, debug=False)

    return predict_api


def load_tests(test_cases_path):
    test_cases_path = os.path.join(test_cases_path)
    with open(test_cases_path, "r") as handle:
        test_cases = yaml.load(handle, Loader=yaml.SafeLoader)
        if isinstance(test_cases, dict):
            return zip(test_cases.keys(), test_cases.values())


def verify_intents(key, predicted_output, expected_output):
    pred_intent_name = predicted_output[const.INTENTS][0][const.NAME]
    true_intent_name = expected_output[const.INTENTS][0][const.NAME]

    assert pred_intent_name == true_intent_name, f"{key} intent name doesn't match!"


def verify_slots(key, predicted_output, expected_output):
    pred_slots = {
        slot[const.NAME]: slot
        for slot in predicted_output[const.INTENTS][0][const.SLOTS]
    }
    true_slots = {
        slot[const.NAME]: slot
        for slot in expected_output[const.INTENTS][0][const.SLOTS]
    }

    assert len(pred_slots) == len(
        true_slots), f"{key} slot-size doesn't match!"
    for pred_slot, true_slot in zip(pred_slots, true_slots):
        pred_slot_values = pred_slots[pred_slot][const.VALUES]
        true_slot_values = true_slots[true_slot][const.VALUES]
        assert len(pred_slot_values) == len(
            true_slot_values
        ), f"{key} - Values for slot={true_slot} doesn't match!"
        for pred_slot_value, true_slot_value in zip(pred_slot_values, true_slot_values):
            assert (
                pred_slot_value[const.TYPE] == true_slot_value[const.TYPE]
            ), f"{key} - Type for slot={true_slot_value[const.TYPE]} doesn't match!"
            assert (
                pred_slot_value[const.VALUE] == true_slot_value[const.VALUE]
            ), f"{key} - Values for slot={true_slot_value[const.TYPE]} doesn't match!"

            # TODO: Add check for range


def verify_entities(key, predicted_output, expected_output):
    pred_entities = {
        entity[const.TYPE]: entity for entity in predicted_output[const.ENTITIES]
    }
    true_entities = {
        entity[const.TYPE]: entity for entity in expected_output[const.ENTITIES]
    }

    assert len(pred_entities) == len(true_entities), (
        f"{key} - number of predicted "
        f"entities does not match number "
        f"of expected entities"
    )

    for expected_entity_type, expected_entity_val in true_entities.items():
        pred_entity_val = pred_entities.get(expected_entity_type)

        assert pred_entity_val, (
            f"{key} - {expected_entity_type} entity "
            f"type not found in predicted entities."
        )
        assert expected_entity_val.get(const.RANGE, {}).get(
            const.START
        ) == pred_entity_val.get(const.RANGE, {}).get(const.START), (
            f"{key} - predicted start index for {expected_entity_type} not "
            f"the same as expected start index."
        )
        assert expected_entity_val.get(const.RANGE, {}).get(
            const.END
        ) == pred_entity_val.get(const.RANGE, {}).get(const.END), (
            f"{key} - predicted end index for {expected_entity_type} not "
            f"the same as expected end index."
        )
        assert expected_entity_val.get(const.PARSERS) == pred_entity_val.get(
            const.PARSERS
        ), (
            f"{key} - predicted parsers for {expected_entity_type} is "
            f"not the same as the expected parsers."
        )
        assert expected_entity_val.get(const.VALUE) == pred_entity_val.get(
            const.VALUE
        ), (
            f"{key} - predicted value for {expected_entity_type} is "
            f"not the same as the expected value."
        )


@pytest.mark.parametrize("key,payload", load_tests("tests/data/test_cases.yaml"))
def test_input_cases(key, payload, slu_api):
    """Test predict_api with inputs."""
    input_ = payload["input"]
    expected = payload["output"]
    output = slu_api(**input_)

    verify_intents(key, output, expected)
    verify_slots(key, output, expected)
    verify_entities(key, output, expected)


def test_classifier_on_training_data(slu_api):
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
    dataset = config.get_dataset(const.CLASSIFICATION, f"{const.TRAIN}.csv")
    test_df = pd.read_csv(dataset).sample(n=100)
    test_df = test_df[~test_df[const.TAG].isin(
        config.tasks.classification.skip)]
    test_df = test_df[test_df[const.ALTERNATIVES] != "[]"]
    test_df = test_df.replace({const.TAG: config.tasks.classification.alias})

    predictions = []
    config.tasks.classification.threshold = 0

    for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        output = slu_api(
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
                const.INTENT_PRED: intents[0][const.NAME] if intents else "_no_preds_",
                const.SCORE: intents[0][const.SCORE] if intents else 0,
            }
        )

    predictions_df = pd.DataFrame(predictions)
    report = make_classification_report(test_df, predictions_df)
    assert report["f1-score"]["weighted avg"] >= 0.9
