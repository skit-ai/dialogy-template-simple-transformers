"""
This module provides a simple interface to provide text features
and receive Intent and Entities.
"""
import operator
import time
import datetime as dt
from datetime import datetime, timedelta
from pprint import pformat
from typing import Any, Dict, List, Optional

import pytz
from dialogy.base import Input, Output
from requests import exceptions

from slu import constants as const
from slu.src.controller.processors import SLUPipeline
from slu.utils import logger
from slu.utils.config import Config


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


def serialize(d):
    if isinstance(d, (dt.date, dt.datetime)):
        return d.isoformat()
    elif isinstance(d, list):
        return [serialize(x) for x in d]
    elif isinstance(d, dict):
        return {key: serialize(val) for key, val in d.items()}
    else:
        return d


def reformat_output(output_dict: Dict[str, Any]) -> Dict[str, Any]:
    intent_slots = output_dict["intents"][0]["slots"]
    output_dict["intents"][0]["slots"] = [slot_val for slot_key, slot_val in intent_slots.items()]
    return serialize(output_dict)


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

        if isinstance(alternatives[0], dict):
            alternatives = [alternatives]

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
            expected_slots=context.get(const.EXPECTED_SLOTS, []),
            nls_label=context.get(const.NLS_LABEL),
        )

        logger.debug(f"Input:\n{pformat(input_)}")
        try:
            _, output = workflow.run(input_)
        except exceptions.ConnectionError as error:
            message = "Could not connect to microservice"
            raise exceptions.ConnectionError(message, error)

        intents = output.intents

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
            output = Output.from_dict(
                {
                    "intents": intents[:1],
                    "entities": output.entities,
                    "original_intent": output.original_intent,
                }
            )

        logger.debug(f"Output:\n{pformat(output.dict())}")
        logger.info(f"Duration: {time.perf_counter() - start_time}s")
        return reformat_output(output.dict())

    return predict
