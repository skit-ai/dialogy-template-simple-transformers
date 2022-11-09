import pytest
import os
import pandas as pd
from tqdm import tqdm
import json
import yaml
from loguru import logger

import argparse
from slu.dev.prompt_setup import setup_prompts
from slu import constants as const

def reset():
    if os.path.exists('config/prompts.yaml'):
        os.remove('config/prompts.yaml')

    if os.path.exists('config/missing_prompts.yaml'):
        os.remove('config/missing_prompts.yaml')

def test_setup_prompts_1():
    reset()
    parser = argparse.ArgumentParser()
    parser.file = 'tests/test_contextual_slu/nls-labels.yaml'
    parser.overwrite = True
    parser.dest = None

    setup_prompts(parser)
    assert os.path.exists('config/prompts.yaml')
    assert os.path.exists('config/missing_prompts.yaml')


def test_setup_prompts_2():
    reset()
    parser = argparse.ArgumentParser()
    parser.file = 'tests/test_contextual_slu/nls-labels.csv'
    parser.overwrite = True
    parser.dest = None
    
    setup_prompts(parser)
    assert os.path.exists('config/prompts.yaml')
    assert os.path.exists('config/missing_prompts.yaml')


def test_setup_prompts_3():
    reset()
    parser = argparse.ArgumentParser()
    parser.overwrite = True
    parser.dest = None
      
    parser.file = 'tests/test_contextual_slu/'
    with pytest.raises(RuntimeError):           
        setup_prompts(parser)

    parser.file = 'tests/test_contextual_slu/nls-labels.txt'
    with pytest.raises(RuntimeError):           
        setup_prompts(parser)

    
    with open('config/prompts.yaml', 'w') as fp:
        pass        
    parser.file = 'tests/test_contextual_slu/nls-labels.yaml'
    parser.overwrite = False
    with pytest.raises(RuntimeError):           
        setup_prompts(parser)
        

test_setup_prompts_1()
test_setup_prompts_2()
test_setup_prompts_3()