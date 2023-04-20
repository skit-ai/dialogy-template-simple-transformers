import pytest
import os
from tqdm import tqdm
import yaml
from slu.utils.config import YamlAliasConfig
from slu import constants as const
import argparse


def load_tests(path)-> dict:
    """
    Custom function to load test cases (.yaml file)
    """
    if not os.path.exists(path):
        raise Exception(
            f"Invalid path: {path}"
        )
    with open(path, 'r') as fp:
        test_cases = yaml.load(fp, Loader=yaml.FullLoader)
    
    return test_cases
        

@pytest.mark.parametrize("test_case", load_tests(path="tests/test_intent_aliasing/data/test_cases.yaml"))
def test_prompt_config(test_case)-> None:
    if test_case['type'] == 'valid_alias':
        """
        Unit test to evaluate Class YAMLPromptConfig from slu.utils.config
        """
        project_config_map = YamlAliasConfig(
                config_path=test_case['args']['file']
            )
        
        if not test_case['args']['is_valid']:
            with pytest.raises(TypeError):
                project_map = project_config_map.generate()
