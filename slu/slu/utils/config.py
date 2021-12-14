"""
[summary]
"""
import abc
import os
import types
from typing import Any, Dict, List, Optional, Set

import attr
import semver
import yaml

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
    datetime_rules = attr.ib(factory=dict, type=Dict[str, Dict[str, Dict[str, int]]], kw_only=True)
    critical_intents = attr.ib(factory=list, type=List[str], kw_only=True)

    def __attrs_post_init__(self) -> None:
        """
        Update default values of attributes from `conifg.yaml`.
        """
        self.tasks = Tasks(**self.tasks)  # type: ignore
        semver.VersionInfo.parse(self.version)
        for purpose in self.tasks.classification.model_args:
            purpose_args = self.tasks.classification.model_args[purpose]
            purpose_args[const.BEST_MODEL_DIR] = self.get_model_dir(
                const.CLASSIFICATION
            )
            purpose_args[const.OUTPUT_DIR] = self.get_model_dir(const.CLASSIFICATION)

    def _get_data_dir(self, task_name: str, version=None) -> str:
        return os.path.join(const.DATA, version or self.version, task_name)

    def get_metrics_dir(self, task_name: str, version=None) -> str:
        return os.path.join(
            self._get_data_dir(task_name, version=version), const.METRICS
        )

    def get_model_dir(self, task_name: str, version=None) -> str:
        return os.path.join(
            self._get_data_dir(task_name, version=version), const.MODELS
        )

    def get_dataset_dir(self, task_name: str, version=None) -> str:
        return os.path.join(
            self._get_data_dir(task_name, version=version), const.DATASETS
        )

    def get_skip_list(self, task_name: str) -> Set[str]:
        if task_name == const.CLASSIFICATION:
            return set(self.tasks.classification.skip)
        raise NotImplementedError(f"Model for {task_name} is not defined!")

    def get_dataset(self, task_name: str, file_name: str, version=None) -> Any:
        return os.path.join(
            self._get_data_dir(task_name, version=version), const.DATASETS, file_name
        )

    def get_model_args(self, task_name: str) -> Dict[str, Any]:
        if task_name == const.CLASSIFICATION:
            return self.tasks.classification.model_args
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
