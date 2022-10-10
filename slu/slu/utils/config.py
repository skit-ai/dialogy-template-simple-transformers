"""
[summary]
"""
import abc
import os
from typing import Any, Dict, List, Optional, Set

import attr
import yaml
import random
from loguru import logger

from slu import constants as const


@attr.s
class Task:
    use = attr.ib(type=bool, kw_only=True, validator=attr.validators.instance_of(bool))
    threshold = attr.ib(
        type=float, kw_only=True, validator=attr.validators.instance_of(float)
    )
    model_args = attr.ib(
        type=dict, kw_only=True, validator=attr.validators.instance_of(dict)
    )
    alias = attr.ib(
        factory=dict, kw_only=True, validator=attr.validators.instance_of(dict)
    )
    skip = attr.ib(
        factory=list, kw_only=True, validator=attr.validators.instance_of(list)
    )
    confidence_levels = attr.ib(
        factory=list, kw_only=True, validator=attr.validators.instance_of(list)
    )
    format = attr.ib(
        factory=str,
        kw_only=True,
        validator=attr.validators.optional(attr.validators.instance_of(str)),
    )


@attr.s
class Tasks:
    classification = attr.ib(
        type=Task, kw_only=True, validator=attr.validators.instance_of(dict)
    )

    def __attrs_post_init__(self) -> None:
        self.classification = Task(**self.classification)  # type: ignore


@attr.s
class Parser:
    params = attr.ib(type=dict, validator=attr.validators.instance_of(dict))
    plugin = attr.ib(type=str, default=None)
    lambda_ = attr.ib(type=str, default=None)

    def __attrs_post_init__(self) -> None:
        error_message = "A Parser should define either a plugin or lambda_ endpoint."
        if isinstance(self.plugin, str) and isinstance(self.lambda_, str):
            raise TypeError(error_message)

        if self.plugin is None and self.lambda_ is None:
            raise TypeError(error_message)


@attr.s
class Config:
    """
    An instance of config handles `config/config.yaml` configurations. This includes reading other related files, models, datasets, etc.

    An instance can:

    - Read config.yaml
    - Modify config.yaml
    - Load models and their configurations
    - Save pickled objects.
    """

    model_name = attr.ib(
        type=str, kw_only=True, validator=attr.validators.instance_of(str)
    )
    version = attr.ib(
        type=str,
        kw_only=True,
        default="0.0.0",
        validator=attr.validators.instance_of(str),
    )
    tasks = attr.ib(type=Tasks, kw_only=True)
    languages = attr.ib(type=List[str], kw_only=True)
    slots: Dict[str, Dict[str, Any]] = attr.ib(factory=dict, kw_only=True)
    calibration = attr.ib(factory=dict, type=Dict[str, Any], kw_only=True)
    entity_patterns = attr.ib(factory=dict, type=Dict[str, List[str]], kw_only=True)
    datetime_rules = attr.ib(
        factory=dict, type=Dict[str, Dict[str, Dict[str, int]]], kw_only=True
    )
    critical_intents = attr.ib(factory=list, type=List[str], kw_only=True)
    timerange_constraints = attr.ib(
        factory=dict, type=Dict[str, Dict[str, Dict[str, Dict[str, int]]]], kw_only=True
    )

    def __attrs_post_init__(self) -> None:
        """
        Update default values of attributes from `conifg.yaml`.
        """
        self.tasks = Tasks(**self.tasks)  # type: ignore
        for purpose in self.tasks.classification.model_args:
            purpose_args = self.tasks.classification.model_args[purpose]
            purpose_args[const.BEST_MODEL_DIR] = self.get_model_dir(
                const.CLASSIFICATION
            )
            purpose_args[const.OUTPUT_DIR] = self.get_model_dir(const.CLASSIFICATION)

    def _get_data_dir(self, task_name: str) -> str:
        return os.path.join(const.DATA, task_name)

    def get_metrics_dir(self, task_name: str) -> str:
        return os.path.join(
            self._get_data_dir(task_name), const.METRICS
        )

    def get_model_dir(self, task_name: str) -> str:
        return os.path.join(
            self._get_data_dir(task_name), const.MODELS
        )

    def get_dataset_dir(self, task_name: str) -> str:
        return os.path.join(
            self._get_data_dir(task_name), const.DATASETS
        )

    def get_skip_list(self, task_name: str) -> Set[str]:
        if task_name == const.CLASSIFICATION:
            return set(self.tasks.classification.skip)
        raise NotImplementedError(f"Model for {task_name} is not defined!")

    def get_dataset(self, task_name: str, file_name: str) -> Any:
        return os.path.join(
            self._get_data_dir(task_name), const.DATASETS, file_name
        )

    def get_model_args(self, task_name: str, purpose: str, **kwargs) -> Dict[str, Any]:
        if task_name == const.CLASSIFICATION:
            args_map = self.tasks.classification.model_args
            if purpose == const.TRAIN:
                if epochs := kwargs.get(const.EPOCHS):
                    args_map[const.TRAIN][const.NUM_TRAIN_EPOCHS] = epochs
            return args_map
        raise NotImplementedError(f"Model for {task_name} is not defined!")

    def get_model_confidence_threshold(self, task_name: str) -> float:
        if task_name == const.CLASSIFICATION:
            return self.tasks.classification.threshold
        raise NotImplementedError(f"Model for {task_name} is not defined!")

    def get_supported_languages(self) -> List[str]:
        return self.languages

    def json(self) -> Dict[str, Any]:
        """
        Represent the class as json

        :return: The class instance as json
        :rtype: Dict[str, Any]
        """
        return attr.asdict(self)

    def save(self) -> None:
        with open(os.path.join("config", "config.yaml"), "w") as handle:
            yaml.dump(self.json(), handle, allow_unicode=True)


class ConfigDataProviderInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate(self):
        ...


class YAMLLocalConfig(ConfigDataProviderInterface):
    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = (
            config_path if config_path else os.path.join("config", "config.yaml")
        )

    def generate(self) -> Dict[str, Config]:
        with open(self.config_path, "r", encoding="utf8") as handle:
            config_dict = yaml.safe_load(handle)
            config = Config(**config_dict)
        return {config_dict[const.MODEL_NAME]: config}


class YAMLPromptConfig(ConfigDataProviderInterface):
    """
    An instance of this class will load, validate and fetch the state <-> prompts mapping from prompts.yaml
    The instance will be further passed to classifier plugins like xlmr.py. 
    """

    def __init__(self, config_path: Optional[str] = None, debug: Optional[bool] = True) -> None:
        self.config_path: str = (
            config_path or const.PROMPTS_CONFIG_PATH
        )
        self.null_prompt_token: str = "<pad>"        
        self.missing_nls_labels: set = set()
        self.config_dict: dict[str] = {}
        self.supported_languages: list[str] = []
        self.debug: bool = debug


    def get_config_dict(self, config_path: str) -> dict:
        error_message = "Unable to read prompts.yaml file, ensure correct format." 
        with open(self.config_path, "r", encoding="utf8") as handle:
            config_dict = yaml.safe_load(handle)
            if not isinstance(config_dict, dict):
                raise TypeError(error_message)
            return config_dict


    def get_supported_languages(self, config_dict: dict) -> List[str]:
        supported_languages: list = list(config_dict.keys())
        return supported_languages

    def _get_config_path(self) -> str:
        return self.config_path

    def _valid_string(self, string: str) -> bool:
        if isinstance(string, str):
            if all([len(string) > 0, string != 'nan', string != ".nan", string != '', string != " ", "Unnamed" not in string]):
                return True
        return False

    def validate(self) -> None:
        if not (all(isinstance(lang,str) for lang in self.config_dict.keys())):
            raise Exception(f"Invalid format, please make sure prompts.yaml is correctly defined")

        for lang in self.config_dict:
            if not (all(isinstance(nls_label, str) for nls_label in self.config_dict[lang]) & all(self._valid_string(nls_label) for nls_label in self.config_dict[lang])):
                raise Exception(f"Invalid or Malformed nls_label name, please make sure prompts.yaml is correctly defined")

            for nls_label in self.config_dict[lang]:
                if not self._valid_string(self.config_dict[lang][nls_label]):
                    raise Exception(f"Invalid or Malformed prompt encountered for nls_label: {nls_label} in lang: {lang}")


    def get_prompt(self, lang: str, nls_label: str, return_all: bool = False) -> List[str]:
        if not lang in self.supported_languages:
            error_message = f"No prompts found for language {lang}, please check the config."            
            raise KeyError(error_message)

        if not nls_label in self.config_dict[lang]:
            self.missing_nls_labels.add(nls_label)
            if self.debug:
                logger.debug(f"NLS Label {nls_label} not found in config for language {lang}")
            return [self.null_prompt_token]

        if not self._valid_string(self.config_dict[lang][nls_label]):
            if self.debug:
                logger.debug(f"No prompt found for NLS Label {nls_label}")
            return [self.null_prompt_token] 
                    
        return self.config_dict[lang][nls_label] if return_all else random.sample(self.config_dict[lang][nls_label], 1)

    def lookup_prompt(self, lang: str, nls_label: str, return_all: bool = False) -> List[str]:
        """
        Same as get_prompt() method, but built for faster lookup to reduce latency during inference. 
        """
        try:
            return random.sample(self.config_dict.get(lang).get(nls_label), 1)
        except Exception as e:
            
            return [self.null_prompt_token]

    def generate(self) -> dict:
        self.config_dict: dict[str] = self.get_config_dict(self.config_path)
        self.supported_languages: list[str] = self.get_supported_languages(self.config_dict)
        self.validate()
        logger.debug(f"List of NLS missing from prompts.yaml:")
        logger.debug(self.missing_nls_labels)
        return self.config_dict


def load_gen_config():
    project_config_map = YAMLLocalConfig().generate()
    return list(project_config_map.values()).pop()


def load_prompt_config(debug=False):
    prompt_config_map = YAMLPromptConfig(debug = debug).generate()
    return prompt_config_map
