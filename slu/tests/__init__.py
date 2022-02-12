import os
import pathlib
import yaml


def load_tests(test_type, current_path):
    test_dir = pathlib.Path(current_path).parent
    test_cases_path = os.path.join(test_dir, f"test_{test_type}.yaml")
    with open(test_cases_path, "r") as handle:
        test_cases = yaml.load(handle, Loader=yaml.SafeLoader)
        if isinstance(test_cases, dict):
            return zip(test_cases.keys(), test_cases.values())
