"""
[summary]
"""
import abc
import os
import re
import pickle
import shutil
from typing import Any, Dict, List, Optional, Union

import attr
import requests
import semver
import torch
import yaml
from requests.adapters import HTTPAdapter
from simpletransformers.classification import ClassificationModel  # type: ignore
from simpletransformers.ner import NERModel
from urllib3.util import Retry

from slu import constants as const
from slu.dev.io.reader.csv import (
    read_ner_dataset_csv,
    save_classification_report,
    save_ner_report,
)
from slu.dev.prepare import prepare
from slu.utils.logger import log
from slu.utils.decorators import task_guard
from slu.utils.error import MissingArtifact
from slu.utils.s3 import get_csvs


@attr.s
class Task:
    use = attr.ib(type=bool, kw_only=True, validator=attr.validators.instance_of(bool))
    threshold = attr.ib(type=float, kw_only=True, validator=attr.validators.instance_of(float))
    model_args = attr.ib(type=dict, kw_only=True, validator=attr.validators.instance_of(dict))
    alias = attr.ib(factory=dict, kw_only=True, validator=attr.validators.instance_of(dict))
    format = attr.ib(
        factory=str, kw_only=True, validator=attr.validators.optional(attr.validators.instance_of(str))
    )

@attr.s
class Tasks:
    classification = attr.ib(type=Task, kw_only=True, validator=attr.validators.instance_of(dict))
    ner = attr.ib(type=Task, kw_only=True, validator=attr.validators.instance_of(dict))

    def __attrs_post_init__(self) -> None:
        self.classification = Task(**self.classification) # type: ignore
        self.ner = Task(**self.ner) # type: ignore


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
    model_name = attr.ib(type=str, kw_only=True, validator=attr.validators.instance_of(str))
    version = attr.ib(type=str, kw_only=True, default="0.0.0", validator=attr.validators.instance_of(str))
    tasks = attr.ib(type=Tasks, kw_only=True)
    languages = attr.ib(type=List[str], kw_only=True)
    slots: Dict[str, Dict[str, Any]] = attr.ib(factory=dict, kw_only=True)
    preprocess: List[Dict[str, Any]] = attr.ib(factory=list, kw_only=True)
    postprocess: List[Dict[str, Any]] = attr.ib(factory=list, kw_only=True)

    def __attrs_post_init__(self) -> None:
        """
        Update default values of attributes from `conifg.yaml`.
        """
        self.tasks = Tasks(**self.tasks) # type: ignore
        semver.VersionInfo.parse(self.version)

    def task_by_name(self, task_name: str) -> Task:
        return getattr(self.tasks, task_name)

    @task_guard
    def get_data_dir(self, task_name: str) -> str:
        return os.path.join(const.DATA, self.version, task_name)

    @task_guard
    def get_metrics_dir(self, task_name: str) -> str:
        return os.path.join(self.get_data_dir(task_name), const.METRICS)

    @task_guard
    def get_model_dir(self, task_name: str, purpose: str) -> str:
        if purpose == const.TRAIN:
            model_dir = os.path.join(self.get_data_dir(task_name), const.MODELS)
        else:
            model_dir = self.task_by_name(task_name).model_args[purpose][const.S_OUTPUT_DIR]
        if not isinstance(model_dir, str):
            raise TypeError(f"Expected model directory for task={task_name}[{purpose}]"
            f" to be a string but {type(model_dir)} was found.")
        return model_dir


    def set_model_dir(self, task: str, purpose:str):

        if purpose == const.TRAIN:

            if task == const.CLASSIFICATION:
                classification_model_dir = self.get_model_dir(const.CLASSIFICATION, const.TRAIN)

                for next_purpose in [const.TEST, const.PRODUCTION.lower()]:

                    self.tasks.classification.model_args[next_purpose][const.S_OUTPUT_DIR] = classification_model_dir
                    self.tasks.classification.model_args[next_purpose][const.S_BEST_MODEL] = classification_model_dir

            elif task == const.NER:
                ner_model_dir = self.get_model_dir(const.NER, const.TRAIN)

                for next_purpose in [const.TEST, const.PRODUCTION.lower()]:

                    self.tasks.ner.model_args[next_purpose][const.S_OUTPUT_DIR] = ner_model_dir
                    self.tasks.ner.model_args[next_purpose][const.S_BEST_MODEL] = ner_model_dir


    @task_guard
    def get_dataset(
        self, task_name: str, purpose: str, file_format=const.CSV, custom_file=None
    ) -> Any:
        data_dir = self.get_data_dir(task_name)
        dataset_dir = os.path.join(data_dir, const.DATASETS)
        dataset_file = custom_file or os.path.join(
            dataset_dir, f"{purpose}.{file_format}"
        )
        alias = self.task_by_name(task_name).alias
        try:
            if task_name == const.CLASSIFICATION:
                data, _ = prepare(
                    dataset_file, alias, file_format=file_format
                )
            elif task_name == const.NER:
                data, _ = read_ner_dataset_csv(dataset_file)
        except FileNotFoundError as file_missing_error:
            raise ValueError(
                f"{dataset_file} not found! Are you sure {const.TASKS}.{task_name}.use = true?"
            ) from file_missing_error

        return data

    @task_guard
    def get_model_args(self, task_name: str, purpose: str) -> Dict[str, Any]:
        model_args = self.task_by_name(task_name).model_args[purpose]
        if not model_args[const.S_OUTPUT_DIR]:
            model_args[const.S_OUTPUT_DIR] = self.get_model_dir(task_name, purpose)

        if not model_args[const.S_BEST_MODEL]:
            model_args[const.S_BEST_MODEL] = self.get_model_dir(task_name, purpose)

        n_epochs = model_args.get(const.S_NUM_TRAIN_EPOCHS)

        eval_batch_size = model_args.get(const.S_EVAL_BATCH_SIZE)

        if purpose == const.TRAIN and not isinstance(n_epochs, int):
            raise TypeError("n_epochs should be an int.")

        if purpose == const.TRAIN and not isinstance(eval_batch_size, int):
            raise TypeError("Number of eval_batch_size should be an int.")

        if n_epochs and purpose == const.TRAIN:
            model_args[const.S_EVAL_DURING_TRAINING_STEPS] = (
                n_epochs * eval_batch_size + const.k
            )

        return model_args

    def get_classification_model(self, purpose: str, labels: List[str]) -> ClassificationModel:
        if not self.tasks.classification.use:
            log.warning(
                "You have set `classification.use = false` within `config.yaml`. Model will not be loaded."
            )
            return None

        model_args = self.get_model_args(const.CLASSIFICATION, purpose)
        kwargs = {
            "use_cuda": ((purpose != const.PRODUCTION.lower()) and (torch.cuda.device_count() > 0)),
            "args": model_args,
        }

        if purpose == const.TRAIN:
            kwargs["num_labels"] = len(labels)

        try:
            return ClassificationModel(
                const.S_XLMR,
                (
                    const.S_XLMRB
                    if purpose == const.TRAIN
                    else model_args[const.S_BEST_MODEL]
                ),
                **kwargs,
            )
        except OSError as os_err:
            raise ValueError(
                f"config/config.yaml has {const.TASKS}.{purpose}.use = True, "
                f"but no model found in {model_args[const.S_OUTPUT_DIR]}"
            ) from os_err

    def get_ner_model(self, purpose: str, labels: List[str]) -> NERModel:
        if not self.tasks.ner.use:
            log.warning(
                "You have set `ner.use = false` within `config.yaml`. Model will not be loaded."
            )
            return None

        model_args = self.get_model_args(const.NER, purpose)

        kwargs = {
            "use_cuda": ((purpose != const.PRODUCTION.lower()) and (torch.cuda.device_count() > 0)),
            "args": model_args,
        }

        if purpose == const.TRAIN:
            kwargs[const.LABELS] = labels

        try:
            return NERModel(
                const.S_XLMR,
                (
                    const.S_XLMRB
                    if purpose == const.TRAIN
                    else model_args[const.S_OUTPUT_DIR]
                ),
                **kwargs,
            )
        except OSError as os_err:
            raise ValueError(
                f"config/config.yaml has {const.TASKS}.{purpose}.use = True, "
                f"but no model found in {model_args[const.S_OUTPUT_DIR]}"
            ) from os_err

    @task_guard
    def get_model(self, task_name: str, purpose: str) -> Union[ClassificationModel, NERModel]:
        labels = self.get_labels(task_name, purpose)
        if task_name == const.NER:
            return self.get_ner_model(purpose, labels)
        return self.get_classification_model(purpose, labels)

    @task_guard
    def get_labels(self, task_name: str, purpose: str) -> List[str]:
        if task_name == const.NER:
            return self.load_pickle(const.NER, purpose, const.S_ENTITY_LABELS)

        try:
            encoder = self.load_pickle(const.CLASSIFICATION, purpose, const.S_INTENT_LABEL_ENCODER)
        except TypeError:
            model_dir = self.get_model_dir(task_name, purpose)
            raise MissingArtifact(const.S_INTENT_LABEL_ENCODER, os.path.join(model_dir, const.S_INTENT_LABEL_ENCODER))
        return encoder.classes_

    @task_guard
    def set_labels(self, task_name: str, purpose: str, labels: List[str]) -> None:
        namespace = (
            const.S_ENTITY_LABELS if task_name == const.NER else const.S_INTENT_LABEL_ENCODER
        )
        self.save_pickle(task_name, purpose, namespace, labels)

    @task_guard
    def save_pickle(self, task_name: str, purpose: str, prop: str, value: Any) -> "Config":
        model_dir = self.get_model_dir(task_name, purpose)
        with open(os.path.join(model_dir, prop), "wb") as handle:
            pickle.dump(value, handle)
        return self

    @task_guard
    def load_pickle(self, task_name: str, purpose: str, prop: str):
        model_dir = self.get_model_dir(task_name, purpose)
        with open(os.path.join(model_dir, prop), "rb") as handle:
            return pickle.load(handle)

    @task_guard
    def get_alias(self, task_name: str) -> Dict[str, str]:
        return self.task_by_name(task_name).alias

    def save_classification_errors(self, df):
        df.to_csv(
            os.path.join(self.get_metrics_dir(const.CLASSIFICATION), const.S_ERRORS),
            index=False,
        )

    def save(self) -> "Config":
        with open(const.S_CONFIG_PATH, "w") as handle:
            yaml.dump(attr.asdict(self), handle, sort_keys=False)
        return self

    @task_guard
    def save_report(self, task, results) -> "Config":
        if task == const.CLASSIFICATION:
            save_classification_report(
                results[0], results[1], self.get_metrics_dir(task)
            )
            return self
        elif task == const.NER:
            save_ner_report(results, self.get_metrics_dir(task))
            return self
        else:
            raise ValueError(
                f"Expected task to be {const.CLASSIFICATION} or {const.NER} instead, {task} was found"
            )

    @task_guard
    def remove_checkpoints(self, task, purpose) -> None:
        model_dir = self.get_model_dir(task, purpose)
        items = os.listdir(model_dir)
        for item in items:
            subdir = os.path.join(model_dir, item)
            if os.path.isdir(subdir):
                shutil.rmtree(subdir)

    def get_supported_languages(self) -> List[str]:
        return self.languages

    def json(self) -> Dict[str, Any]:
        """
        Represent the class as json

        :return: The class instance as json
        :rtype: Dict[str, Any]
        """
        return attr.asdict(self)


class ConfigDataProviderInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate(self):
        ...


class YAMLLocalConfig(ConfigDataProviderInterface):
    def __init__(self, config_path:Optional[str]=None) -> None:
        self.config_path = config_path if config_path else os.path.join("config", "config.yaml")

    def generate(self) -> Dict[str, Config]:
        with open(self.config_path, "r") as handle:
            config_dict = yaml.safe_load(handle)
            config = Config(**config_dict)
        return {config_dict[const.MODEL_NAME]: config}
