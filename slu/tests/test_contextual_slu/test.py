import pytest
import os
from tqdm import tqdm
import yaml
from slu.utils.config import YAMLPromptConfig
from slu.dev.prompt_setup import setup_prompts
from slu import constants as const
import argparse


def load_tests(path='tests/test_contextual_slu/test_cases.yaml')-> dict:
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

def reset_output(dest)-> None:
    """
    Custom function to delete output files. 
    """
    if os.path.exists(os.path.join(dest,'prompts.yaml')):
        os.remove(os.path.join(dest,'prompts.yaml'))
        
    if os.path.exists(os.path.join(dest,'missing_prompts.yaml')):
        os.remove(os.path.join(dest,'missing_prompts.yaml'))
        

def test_prompt_config()-> None:
    """
    Unit tests to evaluate Class YAMLPromptConfig from slu.utils.config
    """
    test_cases = load_tests()
    tests = [test for test in test_cases if test['type'] == 'prompt_config']
    for test in tests:
        project_config_map = YAMLPromptConfig(
                config_path=test['args']['file']
            )
        if test['args']['is_valid']:
            project_map = project_config_map.generate()
            assert isinstance(project_map,dict)
            assert len(project_map.keys()) == 2
            assert(len(project_config_map.supported_languages) == 2)
        else:
            with pytest.raises(TypeError):
                project_map = project_config_map.generate()            
                
                
def test_setup_prompts()-> None:
    """
    Unit tests to evaluate setup_prompts() from slu.dev.setup_prompts
    """
    test_cases = load_tests()
    tests = [test for test in test_cases if test['type'] == 'prompt_setup']
    for test in tests:
        parser = argparse.ArgumentParser()
        parser.file = test['args']['file']
        parser.overwrite = test['args']['overwrite']
        parser.dest = test['args']['dest']
        
        if test['args']['is_valid']:
            reset_output(parser.dest)
            setup_prompts(parser)
            assert os.path.exists(os.path.join(test['args']['dest'],'prompts.yaml'))
            assert os.path.exists(os.path.join(test['args']['dest'],'missing_prompts.yaml'))

        else:
            with pytest.raises(RuntimeError):           
                setup_prompts(parser)