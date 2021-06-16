"""
[summary]
"""
import abc
import os
import pickle
import shutil
from typing import Any, Dict, List, Optional, Union

import attr
import requests
import semver
import yaml
import pydash as py_
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


@attr.s
class Task:
    use = attr.ib(type=bool, kw_only=True, validator=attr.validators.instance_of(bool))
    threshold = attr.ib(type=float, kw_only=True, validator=attr.validators.instance_of(float))
    model_args = attr.ib(type=dict, kw_only=True, validator=attr.validators.instance_of(dict))
    alias = attr.ib(factory=dict, kw_only=True, validator=attr.validators.instance_of(dict))


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
    project_name = attr.ib(type=str, kw_only=True)
    version = attr.ib(type=str, kw_only=True)
    tasks = attr.ib(type=Tasks, kw_only=True)
    preprocess: List[Dict[str, Any]] = attr.ib(factory=list, kw_only=True)
    postprocess: List[Dict[str, Any]] = attr.ib(factory=list, kw_only=True)
    supported_languages: List[str] = attr.ib(factory=list, kw_only=True)

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
    def get_model_dir(self, task_name: str) -> str:
        return self.task_by_name(task_name).model_args[const.S_OUTPUT_DIR]

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
        model_args = self.task_by_name(task_name).model_args
        if not model_args[const.S_OUTPUT_DIR]:
            model_args[const.S_OUTPUT_DIR] = self.get_model_dir(task_name)

        if not model_args[const.S_BEST_MODEL]:
            model_args[const.S_BEST_MODEL] = self.get_model_dir(task_name)

        n_epochs = model_args.get(const.S_NUM_TRAIN_EPOCHS)

        eval_batch_size = model_args.get(const.S_EVAL_BATCH_SIZE)

        if not isinstance(n_epochs, int):
            raise TypeError("n_epochs should be an int.")

        if not isinstance(eval_batch_size, int):
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
            "use_cuda": (purpose != const.PRODUCTION),
            "args": model_args,
        }

        if purpose == const.TRAIN:
            kwargs["num_labels"] = len(labels),

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
            "labels": labels,
            "use_cuda": (purpose != const.PROD),
            "args": model_args,
        }

        if purpose == const.TRAIN:
            kwargs["labels"] = len(labels),

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
        labels = self.get_labels(task_name)
        if task_name == const.NER:
            return self.get_ner_model(purpose, labels)
        return self.get_classification_model(purpose, labels)

    @task_guard
    def get_labels(self, task_name: str) -> List[str]:
        if task_name == const.NER:
            return self.load_pickle(const.NER, const.S_ENTITY_LABELS)

        encoder = self.load_pickle(const.CLASSIFICATION, const.S_INTENT_LABEL_ENCODER)
        return encoder.classes_

    @task_guard
    def set_labels(self, task_name: str, labels: List[str]) -> None:
        namespace = (
            const.S_ENTITY_LABELS if task_name == const.NER else const.S_INTENT_LABEL_ENCODER
        )
        self.save_pickle(task_name, namespace, labels)

    @task_guard
    def save_pickle(self, task_name: str, prop: str, value: Any) -> "Config":
        model_dir = self.get_model_dir(task_name)
        with open(os.path.join(model_dir, prop), "wb") as handle:
            pickle.dump(value, handle)
        return self

    @task_guard
    def load_pickle(self, task_name: str, prop: str):
        model_dir = self.get_model_dir(task_name)
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
    def remove_checkpoints(self, task) -> None:
        model_dir = self.get_model_dir(task)
        items = os.listdir(model_dir)
        for item in items:
            subdir = os.path.join(model_dir, item)
            if os.path.isdir(subdir):
                shutil.rmtree(subdir)

    def get_supported_languages(self) -> List[str]:
        return self.supported_languages

    def find_plugin_metadata(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Access a plugin metadata within _config.

        :param plugin_name: [description]
        :type plugin_name: str
        :return: Value of the sub-property within _config.
        :rtype: Any
        """
        metadata_for_plugins = self.preprocess + self.postprocess
        return py_.find(metadata_for_plugins, lambda plugin_metadata: plugin_metadata.get(const.PLUGIN) == plugin_name)

    def update_plugin_metadata(self, plugin_name: str, param_name: str, value: Any) -> None:
        """
        Update plugin selected params.

        :param plugin_metadata: Metadata object that instantiates a plugin.
        :type plugin_metadata: Optional[Dict[str, Any]]
        :param param_name: The parameter to update.
        :type param_name: str
        :param value: An expected value for the pugin parameter.
        :type value: Any
        """
        plugin_metadata = self.find_plugin_metadata(plugin_name)
        if plugin_metadata:
            plugin_metadata[const.PARAMS][param_name] = value

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


class HTTPConfig(ConfigDataProviderInterface):
    def __init__(self) -> None:
        self.client_configs: Dict[str, Any] = {}

    def _parse_json(self, configs: List[Dict[str, Any]]):
        # if project_configs_response is of List[Dict]
        for config_dict in configs:
            model_name = config_dict.get(const.MODEL_NAME)
            if model_name:
                self.client_configs[model_name] = Config(**config_dict)

    def _get_config(self):
        BUILDER_BACKEND_URL = os.getenv("BUILDER_BACKEND_URL")
        if BUILDER_BACKEND_URL is None:
            raise ValueError(
                f"missing BUILDER_BACKEND_URL env variable, please set it appropriately."
            )

        url = BUILDER_BACKEND_URL + const.CLIENTS_CONFIGS_ROUTE

        session = requests.Session()

        retry = Retry(
            total=const.REQUEST_MAX_RETRIES,
            connect=const.REQUEST_MAX_RETRIES,
            read=const.REQUEST_MAX_RETRIES,
            backoff_factor=0.3,
            status_forcelist=(500, 502, 504)
        )

        http_adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", http_adapter)
        session.mount("https://", http_adapter)

        response = session.get(url, timeout=10)

        if response.ok:
            return response.json()

        raise RuntimeError(f"couldn't establish connection with {url} while trying to collect configs")

    def generate(self) -> Dict[str, Config]:
        if not self.client_configs:
            configs_response = self._get_config()
            self._parse_json(configs_response)
        return self.client_configs


class YAMLLocalConfig(ConfigDataProviderInterface):
    def __init__(self, config_path:Optional[str]=None) -> None:
        self.config_path = config_path if config_path else os.path.join("config", "config.yaml")

    def generate(self) -> Dict[str, Config]:
        with open(self.config_path, "r") as handle:
            config_dict = yaml.load(handle, Loader=yaml.FullLoader)
        return {config_dict[const.PROJECT_NAME]: Config(**config_dict)}
