"""
This module provides a simple interface to provide text features
and receive Intent and Entities.
"""
import os
from typing import Any, Dict, List, Optional

from dialogy import plugins
from dialogy.plugins.preprocess.text.normalize_utterance import normalize

from slu import constants as const
from slu.src.workflow import XLMRWorkflow
from slu.utils.calibration import prepare_calibration_config
from slu.utils.config import Config
from slu.utils.ignore import ignore_utterance
from slu.utils.logger import log
from slu.src.controller.processors import get_preprocessors, get_postprocessors


def predict_wrapper(config_map: Dict[str, Config]):
    """
    Create a closure for the predict function.

    Ensures that the workflow is loaded just once without creating global variables for it.
    This can also be made into a class if needed.
    """
    config = list(config_map.values()).pop()

    workflow = XLMRWorkflow(
        config=config,
        fallback_intent=const.S_INTENT_OOS,
        preprocessors=get_preprocessors(config),
        postprocessors=get_postprocessors(config)
    )

    def predict(
        utterance: Any,
        context: Dict[str, Any],
        intents_info: Optional[List[Dict[str, Any]]] = None,
        reference_time: Optional[int] = None,
        locale: Optional[str] = None,
        lang="en",
    ):
        """
        Produce intent and entities for a given utterance.

        The second argument is context. Use it when available, it is
        a good practice to use it for modeling.
        """
        if ignore_utterance(normalize(utterance)):
            return {
                const.VERSION: config.version,
                const.INTENTS: [{"name": "_oos_"}],
                const.ENTITIES: [],
            }

        utterance = normalize(utterance)

        output = workflow.run(
            {
                const.S_CLASSIFICATION_INPUT: [],
                const.S_CONTEXT: context,
                const.S_INTENTS_INFO: intents_info,
                const.S_NER_INPUT: utterance,
                const.S_REFERENCE_TIME: reference_time,
                const.S_LOCALE: locale,
            }
        )
        intent = output[const.INTENT]
        entities = output[const.ENTITIES]
        workflow.flush()

        intent = intent.json()
        slots = []

        for slot_name, slot_values in intent[const.SLOTS].items():
            slot_values[const.NAME] = slot_name
            slots.append(slot_values)

        intent[const.SLOTS] = slots

        return {
            const.VERSION: config.version,
            const.INTENTS: [intent],
            const.ENTITIES: [entity.json() for entity in entities],
        }

    return predict
