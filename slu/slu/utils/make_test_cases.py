import os
import json
import yaml
import hashlib

import slu.constants as const


def build_test_case(inputs_, outputs) -> None:
    """
    Build a test case from the given input and output data.

    :param inputs_: The input data for the test case.
    :param outputs: The output data for the test case.
    :return: A dictionary representing the test case.
    """
    message = hashlib.sha256()
    if os.environ.get("ENVIRONMENT") == const.PRODUCTION:
        return

    with open(
        os.path.join("tests", "test_controller", "test_cases.yaml"), "r"
    ) as handle:
        test_cases = yaml.load(handle, Loader=yaml.SafeLoader)

    if not test_cases:
        test_cases = []

    message.update(str.encode(json.dumps(inputs_)))
    signature = message.hexdigest()

    if test_cases:
        last_signature = test_cases[-1]["signature"]
        if last_signature == signature:
            return

    with open(
        os.path.join("tests", "test_controller", "test_cases.yaml"), "w"
    ) as handle:
        test_cases.append(
            {
                "input": inputs_,
                "output": json.loads(json.dumps(outputs)),
                "signature": signature,
            }
        )

        yaml.dump(test_cases, handle, default_flow_style=False)
