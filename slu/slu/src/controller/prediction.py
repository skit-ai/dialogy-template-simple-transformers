"""
This module provides a simple interface to provide text features
and receive Intent and Entities.
"""
import copy
import time
from datetime import datetime
from pprint import pformat
from typing import Any, Dict, List, Optional

import pytz
from dialogy.utils import normalize
from dialogy.workflow import Workflow

from slu import constants as const
from slu.src.controller.processors import get_plugins
from slu.utils import logger
from slu.utils.config import Config, YAMLLocalConfig
from slu.utils.make_test_cases import build_test_case


def get_workflow(purpose, **kwargs):
    if "config" in kwargs:
        config = kwargs["config"]
    else:
        project_config_map = YAMLLocalConfig().generate()
        config: Config = list(project_config_map.values()).pop()
    debug = kwargs.get("debug", False)
    return Workflow(get_plugins(purpose, config, debug=debug), debug=debug)


def get_predictions(purpose, **kwargs):
    """
    Create a closure for the predict function.

    Ensures that the workflow is loaded just once without creating global variables for it.
    This can also be made into a class if needed.
    """
    if "config" in kwargs:
        config = kwargs["config"]
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
        start_time = time.perf_counter()
        default_reftime = datetime.now(pytz.timezone("Asia/Kolkata")).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        try:
            reference_time = datetime.fromisoformat(context[const.REFERENCE_TIME])
        except (KeyError, ValueError, TypeError):
            reference_time = default_reftime

        reference_time_as_unix_epoch = int(reference_time.timestamp() * 1000)
        if not lang:
            raise ValueError(f"Expected {lang} to be a ISO-639-1 code.")

        input_ = {
            const.CLASSIFICATION_INPUT: alternatives,
            const.CONTEXT: context,
            const.INTENTS_INFO: intents_info,
            const.NER_INPUT: normalize(alternatives),
            const.REFERENCE_TIME: reference_time_as_unix_epoch,
            const.LOCALE: const.LANG_TO_LOCALES[lang],
        }

        logger.debug(f"Input:\n{pformat(input_)}")
        output = workflow.run(input_=copy.deepcopy(input_))

        intent = output[const.INTENTS][0].json()
        entities = output[const.ENTITIES]

        output = {
            const.VERSION: config.version,
            const.INTENTS: [intent],
            const.ENTITIES: [entity.json() for entity in entities],
        }

        logger.debug(f"Output:\n{output}")
        logger.info(f"Duration: {time.perf_counter() - start_time}s")
        build_test_case(
            {
                "alternatives": alternatives,
                "context": context,
                "lang": lang,
            },
            output,
            **kargs,
        )
        return output

    return predict
