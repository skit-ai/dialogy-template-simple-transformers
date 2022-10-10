import requests
import argparse
from pprint import pprint
import os, yaml, json

from slu import constants as const


def make_request(host_url, payload):
    headers = {"Content-Type": "application/json"}
    response = requests.post(host_url, headers=headers, json=payload)
    return response.json()


def load_tests(test_cases_path):
    test_cases_path = os.path.join(test_cases_path)
    with open(test_cases_path, "r") as handle:
        test_cases = yaml.safe_load(handle)
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

    assert len(pred_slots) == len(true_slots), f"{key} slot-size doesn't match!"
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


def execute_tests(test_cases, args):

    for key, payload in test_cases:
        input_ = payload["input"]
        expected_output = payload["output"]

        response = make_request(args.url, input_)

        verify_intents(key, response.get("response"), expected_output)
        verify_entities(key, response.get("response"), expected_output)
        verify_slots(key, response.get("response"), expected_output)


def main(args):

    test_cases = load_tests(args.path_to_test_cases)
    execute_tests(test_cases, args)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--url", required=True, help="URL endpoint of the deployed service"
    )
    args.add_argument(
        "--path_to_test_cases",
        required=False,
        default="tests/data/test_cases.yaml",
        help="Path to test cases",
    )

    args = args.parse_args()

    main(args)
