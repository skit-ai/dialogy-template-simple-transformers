import pytest
import os
import pandas as pd
from tqdm import tqdm
import json
import yaml

from slu.utils.config import YAMLPromptConfig
from slu import constants as const

def test_prompt_config_1():

    project_config_map = YAMLPromptConfig(
        config_path="tests/test_contextual_slu/test_case_1.yaml"
    )
    project_map = project_config_map.generate()
    assert isinstance(project_map,dict)
    assert len(project_map.keys()) == 2
    assert(len(project_config_map.supported_languages) == 2)


def test_prompt_config_2():

    project_config_map = YAMLPromptConfig(
        config_path="tests/test_contextual_slu/test_case_2.yaml"
    )
    
    with pytest.raises(TypeError):
        project_map = project_config_map.generate()


def test_prompt_config_3():

    project_config_map = YAMLPromptConfig(
        config_path="tests/test_contextual_slu/test_case_3.yaml"
    )
    
    with pytest.raises(TypeError):
        project_map = project_config_map.generate()


def test_prompt_config_4():

    project_config_map = YAMLPromptConfig(
        config_path="tests/test_contextual_slu/test_case_4.yaml"
    )
    
    with pytest.raises(TypeError):
        project_map = project_config_map.generate()
        

def test_prompt_config_5():

    project_config_map = YAMLPromptConfig(
        config_path="tests/test_contextual_slu/test_case_5.yaml"
    )
    
    with pytest.raises(TypeError):
        project_map = project_config_map.generate()    
        

test_prompt_config_1()
test_prompt_config_2()
test_prompt_config_3()
test_prompt_config_4()
test_prompt_config_5()