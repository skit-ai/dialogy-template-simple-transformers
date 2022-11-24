import pytest
import os
from tqdm import tqdm
import yaml
from slu.utils.config import YAMLPromptConfig
from slu.dev.prompt_setup import setup_prompts, fill_nls_col
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

def reset_output(dest)-> None:
    """
    Custom function to delete output files. 
    """
    if os.path.exists(os.path.join(dest,'prompts.yaml')):
        os.remove(os.path.join(dest,'prompts.yaml'))
        
    if os.path.exists(os.path.join(dest,'missing_prompts.yaml')):
        os.remove(os.path.join(dest,'missing_prompts.yaml'))
        

@pytest.mark.parametrize("test_case", load_tests(path="tests/test_contextual_slu/test_cases.yaml"))
def test_prompt_config(test_case)-> None:
    if test_case['type'] == 'prompt_config':
        """
        Unit test to evaluate Class YAMLPromptConfig from slu.utils.config
        """
        project_config_map = YAMLPromptConfig(
                config_path=test_case['args']['file']
            )
        if test_case['args']['is_valid']:
            project_map = project_config_map.generate()
            assert isinstance(project_map,dict)
            assert len(project_map.keys()) == 2
            assert(len(project_config_map.supported_languages) == 2)
        else:
            with pytest.raises(TypeError):
                project_map = project_config_map.generate()

    elif test_case['type'] == 'prompt_setup':
        """
        Unit tests to evaluate setup_prompts() from slu.dev.setup_prompts
        """
        parser = argparse.ArgumentParser()
        parser.file = test_case['args']['file']
        parser.overwrite = test_case['args']['overwrite']
        parser.dest = test_case['args']['dest']
        
        if test_case['args']['is_valid']:
            reset_output(parser.dest)
            setup_prompts(parser)
            assert os.path.exists(os.path.join(test_case['args']['dest'],'prompts.yaml'))
            assert os.path.exists(os.path.join(test_case['args']['dest'],'missing_prompts.yaml'))

        else:
            with pytest.raises(RuntimeError):           
                setup_prompts(parser)
                
    elif test_case['type'] == 'fill_nls_col':
        """
        Unit tests to evaluate setup_prompts() from slu.dev.setup_prompts
        """
        parser = argparse.ArgumentParser()
        parser.input_file = test_case['args']['input_file']
        parser.overwrite = test_case['args']['overwrite']
        parser.output_file = test_case['args']['output_file']
        
        if not test_case['args']['is_valid']:
            with pytest.raises(RuntimeError):           
                fill_nls_col(parser)
