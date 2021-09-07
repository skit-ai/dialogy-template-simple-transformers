import hashlib
import json
import os

import yaml

import slu.constants as const


def build_test_case(inputs_, outputs, ignore_test_case=False) -> None:
    """
    Build a test case from the given input and output data.

    :param inputs_: The input data for the test case.
    :param outputs: The output data for the test case.
    :return: A dictionary representing the test case.
    """
    if ignore_test_case:
        return None
    message = hashlib.sha256()
    if os.environ.get("ENVIRONMENT") == const.PRODUCTION:
        return

    with open(
        os.path.join("tests", "test_controller", "test_cases.yaml"), "r"
    ) as handle:
        test_cases = yaml.load(handle, Loader=yaml.SafeLoader)

    if not test_cases:
        test_cases = {}

    message.update(str.encode(json.dumps(inputs_)))
    signature = message.hexdigest()

    if test_cases:
        if signature in test_cases:
            return None

    with open(
        os.path.join("tests", "test_controller", "test_cases.yaml"), "w"
    ) as handle:
        test_cases[signature] = {
            "input": inputs_,
            "output": json.loads(json.dumps(outputs)),
        }

        yaml.dump(test_cases, handle, default_flow_style=False)
