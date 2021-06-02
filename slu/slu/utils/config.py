"""
[summary]
"""
import abc
import os
import pickle
import shutil
import traceback
from typing import Any, Dict, List, Optional


import attr
import requests
import semver
import yaml
from simpletransformers.classification import ClassificationModel  # type: ignore
from simpletransformers.ner import NERModel

from slu import constants as const
from slu.dev.io.reader.csv import (
    read_ner_dataset_csv,
    save_classification_report,
    save_ner_report,
)
from slu.dev.prepare import prepare
from slu.utils.logger import log



class ConfigDataProviderInterface(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def give_config_data(self):
        return


class YAMLLocalConfigDataProvider(ConfigDataProviderInterface):

    def __init__(self, config_path:Optional[str]=None) -> None:
        super().__init__()
        self.config = None
        self.config_path = config_path if config_path else os.path.join("config", "config.yaml")



    def give_config_data(self) -> Dict:

        if self.config is None:
            with open(self.config_path, "r") as handle:
                self.config = yaml.load(handle, Loader=yaml.FullLoader)
        return self.config


class JSONAPIConfigDataProvider(ConfigDataProviderInterface):

    def __init__(self) -> None:
        super().__init__()
        self.config = None

    def _get_config_from_api(self):
        url = os.getenv("BUILDER_BACKEND_URL")
        for _ in range(5):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    break
                log.warning(f"Call to Builder backend failed. Trying again")
            except Exception:
                log.error(traceback.format_exc())
        self.config = response.json()


    def update_config(self, config_dict):
        self.config = config_dict

    def give_config_data(self) -> Dict:
        if self.config is None:
            self._get_config_from_api()
        return self.config




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

    # **config_path**
    #
    # In case config/config.yaml is not the appropriate location.
    config_path = attr.ib(default=os.path.join("config", "config.yaml"))

    # **project_name**
    #
    # The name of your project.
    project_name = attr.ib(default=None)

    # **_config**
    #
    # Read contents of `config.yaml` into a Dict.
    _config = attr.ib(default=None)

    # **version**
    #
    # Current project version.
    version = attr.ib(default=None)

    # **n_cores**
    #
    # Number of cpu cores to use for certain jobs.
    n_cores = attr.ib(default=None)

    # **classification_threshold**
    #
    # Confidence threshold for classification tasks.
    classification_threshold = attr.ib(default=None)

    # **ner_threshold**
    #
    # Confidence threshold for NER tasks.
    ner_threshold = attr.ib(default=None)

    # **rules**
    #
    # Rules for filling Entities within Intent slots.
    rules = attr.ib(default=None)

    # **config_data_provider**
    #
    # dictionary source if we are getting config from API.
    config_data_provider : ConfigDataProviderInterface = attr.ib(default=YAMLLocalConfigDataProvider())

    def __attrs_post_init__(self) -> None:
        """
        Update default values of attributes from `conifg.yaml`.
        """
        self._config = self.config_data_provider.give_config_data()
        semver.VersionInfo.parse(self._config.get(const.VERSION))
        self.version = self._config.get(const.VERSION)
        self.n_cores = self._config.get(const.CORES)
        self.classification_threshold = self._config[const.TASKS][const.CLASSIFICATION][
            const.THRESHOLD
        ]

        self.classification_file_format = self._config[const.TASKS][
            const.CLASSIFICATION
        ][const.FORMAT]

        self.rules = self._config[const.RULES]
        self.use_classifier = self._config[const.TASKS][const.CLASSIFICATION][const.USE]

        self.ner_file_format = self._config[const.TASKS][const.NER][const.FORMAT]

        self.ner_threshold = self._config[const.TASKS][const.NER][const.THRESHOLD]
        self.use_ner = self._config[const.TASKS][const.NER][const.USE]

        self.supported_tasks = []
        if self.use_classifier:
            self.supported_tasks.append(const.CLASSIFICATION)
        if self.use_ner:
            self.supported_tasks.append(const.NER)

    def set_version(self, version) -> "Config":
        self.version = version
        self._config[const.VERSION] = version
        return self

    def get_data_dir(self, task) -> str:
        return os.path.join(const.DATA, self.version, task)

    def get_metrics_dir(self, task) -> str:
        return os.path.join(self.get_data_dir(task), const.METRICS)

    def get_model_dir(self, task) -> str:
        return os.path.join(self.get_data_dir(task), const.MODELS)

    def get_model_path(self, task) -> str:
        return os.path.join(self.get_model_dir(task), self.version)

    def get_dataset(
        self, task, purpose, file_format=const.CSV, custom_file=None
    ) -> Any:
        error_message = (
            f"Expected task to be in {self.supported_tasks} instead, {task} was found!"
        )

        data_dir = os.path.join(const.DATA, self.version, task)
        dataset_dir = os.path.join(data_dir, const.DATASETS)
        dataset_file = custom_file or os.path.join(
            dataset_dir, f"{purpose}.{file_format}"
        )
        alias = self.get_alias(task)

        try:
            if task == const.CLASSIFICATION:
                data, _ = prepare(
                    dataset_file, alias, n_cores=self.n_cores, file_format=file_format
                )
            elif task == const.NER:
                data, labels = read_ner_dataset_csv(dataset_file)
            else:
                raise ValueError(error_message)
        except FileNotFoundError as file_missing_error:
            raise ValueError(
                f"{dataset_file} not found! Are you sure {const.TASKS}.{purpose}.use = true?"
                " within config/config.yaml?"
            ) from file_missing_error

        return data

    def get_model_args(self, task, purpose) -> Dict[str, Any]:
        model_args = self._config[const.TASKS][task][const.S_MODEL_ARGS][purpose]
        model_args[const.S_OUTPUT_DIR] = self.get_model_dir(task)
        model_args[const.S_BEST_MODEL] = self.get_model_dir(task)
        n_epochs = model_args.get(const.S_NUM_TRAIN_EPOCHS)
        eval_batch_size = model_args.get(const.S_EVAL_BATCH_SIZE)

        if purpose == const.TRAIN:
            model_args[const.S_EVAL_DURING_TRAINING_STEPS] = (
                n_epochs * eval_batch_size + const.k
            )

        return model_args

    def get_classification_model(
        self, purpose, labels
    ) -> Optional[ClassificationModel]:
        if not self.use_classifier:
            log.warning(
                "You have set `classification.use = false` within `config.yaml`. Model will not be loaded."
            )
            return None

        model_args = self.get_model_args(const.CLASSIFICATION, purpose)
        kwargs = {
            "num_labels": len(labels),
            "use_cuda": (purpose != const.PROD),
            "args": model_args,
        }
        if purpose != const.TRAIN:
            del kwargs["num_labels"]

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

    def get_ner_model(self, purpose, labels) -> Optional[NERModel]:
        if not self.use_ner:
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
        if purpose != const.TRAIN:
            del kwargs["labels"]

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

    def get_labels(self, task) -> List[str]:
        if task == const.CLASSIFICATION:
            encoder = self.load_pickle(
                const.CLASSIFICATION, const.S_INTENT_LABEL_ENCODER
            )
            return encoder.classes_
        elif task == const.NER:
            return self.load_pickle(const.NER, const.S_ENTITY_LABELS)
        else:
            raise ValueError(
                f"Expected task to be {const.CLASSIFICATION} or {const.NER} instead, {task} was found"
            )

    def set_labels(self, task, labels) -> None:
        if task not in self.supported_tasks:
            raise ValueError(f"Expected task to be one of {self.supported_tasks}")
        namespace = (
            const.S_ENTITY_LABELS if task == const.NER else const.S_INTENT_LABEL_ENCODER
        )
        self.save_pickle(task, namespace, labels)

    def get_alias(self, task) -> List[str]:
        return self._config.get(const.TASKS, {}).get(task, {}).get(const.ALIAS)

    def save_classification_errors(self, df):
        df.to_csv(
            os.path.join(self.get_metrics_dir(const.CLASSIFICATION), const.S_ERRORS),
            index=False,
        )

    def get_model(self, task, purpose):
        labels = self.get_labels(task)
        error_message = (
            f"Expected task to be in {self.supported_tasks} instead, {task} was found!"
        )

        if task == const.CLASSIFICATION:
            return self.get_classification_model(purpose, labels)
        elif task == const.NER:
            return self.get_ner_model(purpose, labels)
        else:
            raise ValueError(error_message)

    def save_pickle(self, task, prop, value) -> "Config":
        model_dir = self.get_model_dir(task)
        with open(os.path.join(model_dir, prop), "wb") as handle:
            pickle.dump(value, handle)
        return self

    def load_pickle(self, task, prop):
        model_dir = self.get_model_dir(task)
        with open(os.path.join(model_dir, prop), "rb") as handle:
            return pickle.load(handle)

    def save(self) -> "Config":
        with open(self.config_path, "w") as handle:
            yaml.dump(self._config, handle, sort_keys=False)
        return self

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

    def remove_checkpoints(self, task) -> None:
        model_dir = self.get_model_dir(task)
        items = os.listdir(model_dir)
        for item in items:
            subdir = os.path.join(model_dir, item)
            if os.path.isdir(subdir):
                shutil.rmtree(subdir)

    def get_supported_langauges(self) -> Dict[str, str]:
        return self._config["languages"]
