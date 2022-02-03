"""
This module provides a simple interface to provide text features
and receive Intent and Entities.
"""
import os
import copy
import time
import operator
from requests import exceptions
from datetime import datetime, timedelta
from pprint import pformat
from typing import Any, Dict, List, Optional

import pytz
from dialogy.utils import normalize
from dialogy.workflow import Workflow
from dialogy.types import Intent

from slu import constants as const
from slu.src.controller.processors import get_plugins
from slu.utils import logger
from slu.utils.config import Config, YAMLLocalConfig
from slu.utils.make_test_cases import build_test_case


def get_workflow(purpose, **kwargs):
    if const.CONFIG in kwargs:
        config = kwargs[const.CONFIG]
    else:
        project_config_map = YAMLLocalConfig().generate()
        config: Config = list(project_config_map.values()).pop()
    debug = kwargs.get("debug", False)
    return Workflow(get_plugins(purpose, config, debug=debug), debug=debug)


def get_reftime(config: Config, context: Dict[str, Any], lang: str):
    default_reftime = datetime.now(pytz.timezone("Asia/Kolkata")).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    try:
        reference_time = datetime.fromisoformat(context[const.REFERENCE_TIME])
    except (KeyError, ValueError, TypeError):
        reference_time = default_reftime

    current_state = context.get(const.CURRENT_STATE)

    if current_state in config.datetime_rules:
        if const.REWIND not in config.datetime_rules[current_state] and const.FORWARD not in config.datetime_rules[current_state]:
            raise NotImplementedError(f"Expected either {const.FORWARD} or {const.REWIND} in {config.datetime_rules}")

        if const.REWIND in config.datetime_rules[current_state]:
            operation = operator.sub
            kwargs = config.datetime_rules[current_state][const.REWIND]
        elif const.FORWARD in config.datetime_rules[current_state]:
            operation = operator.add
            kwargs = config.datetime_rules[current_state][const.FORWARD]
        reference_time = operation(reference_time, timedelta(**kwargs))

    return int(reference_time.timestamp() * 1000)


def get_predictions(purpose, **kwargs):
    """
    Create a closure for the predict function.

    Ensures that the workflow is loaded just once without creating global variables for it.
    This can also be made into a class if needed.
    """
    if const.CONFIG in kwargs:
        config = kwargs[const.CONFIG]
    else:
        project_config_map = YAMLLocalConfig().generate()
        config: Config = list(project_config_map.values()).pop()
    workflow = get_workflow(purpose, **kwargs)

    def predict(
        alternatives: Any,
        context: Optional[Dict[str, Any]] = None,
        intents_info: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[Any]] = None,
        lang: Optional[str] = None,
        **kargs,
    ):
        """
        Produce intent and entities for a given utterance.

        The second argument is context. Use it when available, it is
        a good practice to use it for modeling.
        """
        context = context or {}
        history = history or []
        if not lang:
            raise ValueError(f"Expected {lang} to be a ISO-639-1 code.")

        start_time = time.perf_counter()
        reference_time_as_unix_epoch = get_reftime(config, context, lang)

        input_ = {
            const.CLASSIFICATION_INPUT: alternatives,
            const.CONTEXT: context,
            const.INTENTS_INFO: intents_info,
            const.NER_INPUT: normalize(alternatives),
            const.REFERENCE_TIME: reference_time_as_unix_epoch,
            const.LOCALE: const.LANG_TO_LOCALES[lang],
        }

        logger.debug(f"Input:\n{pformat(input_)}")
        try:
            output = workflow.run(input_=copy.deepcopy(input_))
        except exceptions.ConnectionError as error:
            if os.environ.get("ENVIRONMENT") == const.PRODUCTION:
                message = "Could not connect to duckling."
            else:
                message = "Could not connect to duckling. If you don't need duckling then it seems safe to remove it in this environment."
            raise exceptions.ConnectionError(message) from error

        intents: List[Intent] = output[const.INTENTS]
        intents_json = [intent.json() for intent in output[const.INTENTS]]
        entities = output[const.ENTITIES]

        confidence_levels = config.tasks.classification.confidence_levels
        if confidence_levels:
            for idx in range(len(intents)):
                low, high = confidence_levels
                if intents[idx].score <= low:
                    intents_json[idx][const.CONFIDENCE_LEVEL] = const.LOW
                elif intents[idx].score <= high:
                    intents_json[idx][const.CONFIDENCE_LEVEL] = const.MEDIUM
                else: 
                    intents_json[idx][const.CONFIDENCE_LEVEL] = const.HIGH

        output = {
            const.VERSION: config.version,
            const.INTENTS: intents_json,
            const.ENTITIES: [entity.json() for entity in entities],
        }

        logger.debug(f"Output:\n{output}")
        logger.info(f"Duration: {time.perf_counter() - start_time}s")
        build_test_case(
            {
                const.ALTERNATIVES: alternatives,
                const.CONTEXT: context,
                const.LANG: lang,
            },
            output,
            **kargs,
        )
        return output

    return predict
