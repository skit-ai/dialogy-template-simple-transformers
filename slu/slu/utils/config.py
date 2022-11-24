"""
[summary]
"""
import abc
import os
import random
from typing import Any, Dict, List, Optional, Set

import attr
import yaml
from loguru import logger

from slu import constants as const
from slu.utils.validations import valid_string


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
        return os.path.join(self._get_data_dir(task_name), const.METRICS)

    def get_model_dir(self, task_name: str) -> str:
        return os.path.join(self._get_data_dir(task_name), const.MODELS)

    def get_dataset_dir(self, task_name: str) -> str:
        return os.path.join(self._get_data_dir(task_name), const.DATASETS)

    def get_skip_list(self, task_name: str) -> Set[str]:
        if task_name == const.CLASSIFICATION:
            return set(self.tasks.classification.skip)
        raise NotImplementedError(f"Model for {task_name} is not defined!")

    def get_dataset(self, task_name: str, file_name: str) -> Any:
        return os.path.join(self._get_data_dir(task_name), const.DATASETS, file_name)

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
    An instance of this class can:
    -   Load and validate config/prompts.yaml.
    -   Store information from config/prompts.yaml into an object (prompt_config_map). 
        prompt_config_map maps nls_label -> prompt. 
        Text classification plugins like XLMR will input prompt_config_map.
    """

    def __init__(
        self, config_path: Optional[str] = None, null_prompt_token: Optional[str] = const.PROMPT_NOISE_FILLER_TOKEN,debug: Optional[bool] = True
    ) -> None:
        self.config_path: str = config_path or const.PROMPTS_CONFIG_PATH
        self.null_prompt_token: str = null_prompt_token
        self.missing_nls_labels: set = set()
        self.config_dict: dict[str] = {}
        self.supported_languages: list[str] = []
        self.debug: bool = debug

    def get_config_dict(self) -> dict:
        with open(self.config_path, "r", encoding="utf8") as handle:
            config_dict = yaml.safe_load(handle)
            if not isinstance(config_dict, dict):
                raise TypeError("Unable to read prompts.yaml file, ensure correct format.")
            return config_dict

    def get_config_path(self) -> str:
        return self.config_path

    def validate(self) -> None:
        if not (all(isinstance(lang, str) for lang in self.config_dict.keys())):
            raise TypeError(
                f"Invalid format, please make sure prompts.yaml is correctly defined"
            )

        for lang in self.config_dict:
            if not (
                all(isinstance(nls_label, str) for nls_label in self.config_dict[lang])
                & all(
                    valid_string(nls_label)
                    for nls_label in self.config_dict[lang]
                )
            ):
                raise TypeError(
                    f"Invalid or Malformed nls_label name, please make sure prompts.yaml is correctly defined"
                )

            for nls_label in self.config_dict[lang]:
                if not valid_string(self.config_dict[lang][nls_label]):
                    raise TypeError(
                        f"Invalid or Malformed prompt encountered for nls_label: {nls_label} in lang: {lang}"
                    )

    def generate(self) -> dict:
        """
        Create, validate, and return a dictionary mapping between nls_labels and their respective prompts. 
        :rtype: Dict[str, str]
        """
        self.config_dict: Dict[str] = self.get_config_dict()
        self.supported_languages: list = list(self.config_dict.keys())
        self.validate()      
        if self.debug:
            logger.debug(f"Found following NLS Labels missing in config/prompts.yaml:")
            logger.debug(self.missing_nls_labels)
            
        return self.config_dict


def load_gen_config():
    project_config_map = YAMLLocalConfig().generate()
    return list(project_config_map.values()).pop()


def load_prompt_config(debug=False):
    prompt_config_map = YAMLPromptConfig(debug=debug).generate()
    return prompt_config_map
