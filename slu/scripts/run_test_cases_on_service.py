import requests
import argparse
import os, yaml

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


def execute_tests(test_cases, args):

    for key, payload in test_cases:
        input_ = payload["input"]
        output = payload["output"]

        response = make_request(args.url, input_)

        pred_intent_name = response["response"][const.INTENTS][0][const.NAME]
        true_intent_name = output[const.INTENTS][0][const.NAME]

        assert pred_intent_name == true_intent_name, f"{key} intent name doesn't match!"


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
