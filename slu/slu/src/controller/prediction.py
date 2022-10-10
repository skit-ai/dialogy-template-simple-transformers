"""
This module provides a simple interface to provide text features
and receive Intent and Entities.
"""
import os
import copy
import time
import pytz
import operator
from requests import exceptions
from datetime import datetime, timedelta
from pprint import pformat
from typing import Any, Dict, List, Optional

from dialogy.base import Input

from slu import constants as const
from slu.src.controller.processors import SLUPipeline
from slu.utils import logger
from slu.utils.config import Config, YAMLLocalConfig
from slu.utils.make_test_cases import build_test_case


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
        if (
            const.REWIND not in config.datetime_rules[current_state]
            and const.FORWARD not in config.datetime_rules[current_state]
        ):
            raise NotImplementedError(
                f"Expected either {const.FORWARD} or {const.REWIND} in {config.datetime_rules}"
            )

        if const.REWIND in config.datetime_rules[current_state]:
            operation = operator.sub
            kwargs = config.datetime_rules[current_state][const.REWIND]
        elif const.FORWARD in config.datetime_rules[current_state]:
            operation = operator.add
            kwargs = config.datetime_rules[current_state][const.FORWARD]
        reference_time = operation(reference_time, timedelta(**kwargs))

    return int(reference_time.timestamp() * 1000)


def get_predictions(purpose, final_plugin=None, **kwargs):
    """
    Create a closure for the predict function.

    Ensures that the workflow is loaded just once without creating global variables for it.
    This can also be made into a class if needed.
    """
    pipeline = SLUPipeline(**kwargs)
    workflow = pipeline.get_workflow(purpose, final_plugin)

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
            logger.info(f"Expected {lang=} to be a ISO-639-1 code.")
            logger.info("setting default lang=hi since no lang was provided")
            lang = "hi"

        start_time = time.perf_counter()
        reference_time_as_unix_epoch = get_reftime(pipeline.config, context, lang)

        input_ = Input(
            utterances=alternatives,
            reference_time=reference_time_as_unix_epoch,
            locale=const.LANG_TO_LOCALES[lang],
            lang=lang,
            slot_tracker=intents_info,
            timezone="Asia/Kolkata",
            current_state=context.get(const.CURRENT_STATE),
            previous_intent=context.get(const.CURRENT_INTENT),
            expected_slots=context.get(const.EXPECTED_SLOTS),
            nls_label=context.get(const.NLS_LABEL)
        )

        logger.debug(f"Input:\n{pformat(input_)}")
        try:
            _, output = workflow.run(input_)
        except exceptions.ConnectionError as error:
            if os.environ.get("ENVIRONMENT") == const.PRODUCTION:
                message = "Could not connect to duckling."
            else:
                message = "Could not connect to duckling. If you don't need duckling then it seems safe to remove it in this environment."
            raise exceptions.ConnectionError(message) from error

        intents = output.get(const.INTENTS, [])

        confidence_levels = pipeline.config.tasks.classification.confidence_levels

        if confidence_levels:
            for intent in intents:
                low, high = confidence_levels
                if intent[const.SCORE] <= low:
                    intent[const.CONFIDENCE_LEVEL] = const.LOW
                elif intent[const.SCORE] <= high:
                    intent[const.CONFIDENCE_LEVEL] = const.MEDIUM
                else:
                    intent[const.CONFIDENCE_LEVEL] = const.HIGH

        if intents and purpose == const.PRODUCTION:
            output[const.INTENTS] = intents[:1]

        logger.debug(f"Output:\n{output}")
        logger.info(f"Duration: {time.perf_counter() - start_time}s")
        return output

    return predict
