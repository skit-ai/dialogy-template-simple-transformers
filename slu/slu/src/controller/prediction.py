"""
This module provides a simple interface to provide text features
and receive Intent and Entities.
"""
from typing import Any, Dict, List, Optional

from dialogy.parser.text.entity.duckling_parser import DucklingParser
from dialogy.postprocess.text.slot_filler.rule_slot_filler import (
    RuleBasedSlotFillerPlugin,
)
from dialogy.types.entity import BaseEntity
from dialogy.preprocess.text.merge_asr_output import merge_asr_output_plugin
from dialogy.preprocess.text.normalize_utterance import normalize

from slu import constants as const
from slu.src.workflow import XLMRWorkflow
from slu.utils.config import Config

config = Config()

slot_filler = RuleBasedSlotFillerPlugin(
    rules=config.rules[const.SLOTS], access=lambda w: w.output
)()


def update_input(w: XLMRWorkflow, value: str) -> None:
    w.input[const.S_CLASSIFICATION_INPUT] = value


merge_asr_output = merge_asr_output_plugin(
    access=lambda w: w.input[const.S_CLASSIFICATION_INPUT], mutate=update_input
)


def update_entities(workflow: XLMRWorkflow, entities: List[BaseEntity]):
    intents, collected_entities = workflow.output
    workflow.output = (intents, collected_entities + entities)


duckling_parser = DucklingParser(
    access=lambda w: (
        w.input[const.S_CLASSIFICATION_INPUT],
        w.input[const.S_REFERENCE_TIME],
        w.input[const.S_LOCALE]
    ),
    mutate=update_entities,
    dimensions=["number"],
    locale="en_IN",
    timezone="Asia/Kolkata",
)()


def predict_wrapper():
    """
    Create a closure for the predict function.

    Ensures that the workflow is loaded just once without creating global variables for it.
    This can also be made into a class if needed.
    """
    preprocessors = [
        merge_asr_output,
        duckling_parser,
    ]

    postprocessors = [
        slot_filler
        # slot_filler should always be the last plugin.
        # If you add entities afterwards, they wont fill intent slots.
    ]

    workflow = XLMRWorkflow(
        preprocessors=preprocessors,
        postprocessors=postprocessors,
    )

    def predict(
        utterance: List[str],
        context: Dict[str, Any],
        intents_info: Optional[List[Dict[str, Any]]] = None,
        reference_time: Optional[int] = None,
        locale: Optional[str] = None
    ):
        """
        Produce intent and entities for a given utterance.

        The second argument is context. Use it when available, it is
        a good practice to use it for modeling.
        """
        utterance = normalize(utterance)

        intent, entities = workflow.run(
            {
                const.S_CLASSIFICATION_INPUT: utterance,
                const.S_CONTEXT: context,
                const.S_INTENTS_INFO: intents_info,
                const.S_NER_INPUT: utterance,
                const.S_REFERENCE_TIME: reference_time,
                const.S_LOCALE: locale
            }
        )
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
