"""
This module provides a simple interface to provide text features
and receive Intent and Entities.
"""
import importlib
import traceback
from typing import Any, Dict, List, Optional
from dialogy.plugins.preprocess.text.normalize_utterance import normalize
from dialogy import plugins

from slu import constants as const
from slu.src.workflow import XLMRWorkflow
from slu.utils.config import Config
from slu.dev.plugin_parse import plugin_functions
from slu.utils.logger import log


merge_asr_output = plugins.MergeASROutputPlugin(
    access=plugin_functions.access(const.INPUT, const.S_CLASSIFICATION_INPUT),
    mutate=plugin_functions.mutate(const.INPUT, const.S_CLASSIFICATION_INPUT),
)()

duckling_plugin = plugins.DucklingPlugin(
    access=plugin_functions.access(const.INPUT, const.S_NER_INPUT, const.S_REFERENCE_TIME, const.S_LOCALE),
    mutate=plugin_functions.mutate(const.OUTPUT, const.ENTITIES),
    dimensions=["people", "number", "time", "duration"],
    locale="en_IN",
    timezone="Asia/Kolkata",
    timeout=0.5,
    # Works only in development mode. You need to set this in k8s configs.
    url=os.environ.get("DUCKLING_URL", "http://localhost:8000/parse/"),
)()


def predict_wrapper(config_map: Dict[str, Config]):
    """
    Create a closure for the predict function.

    Ensures that the workflow is loaded just once without creating global variables for it.
    This can also be made into a class if needed.
    """
    config: Config = list(config_map.values()).pop()
    slot_filler = plugins.RuleBasedSlotFillerPlugin(
        access=plugin_functions.access(const.OUTPUT, const.INTENT, const.ENTITIES),
        rules=config.slots,
    )()

    preprocessors = [merge_asr_output, duckling_plugin]
    postprocessors = [slot_filler]

    workflow = XLMRWorkflow(
        preprocessors=preprocessors,
        postprocessors=postprocessors,
        fallback_intent="_oos_",
        config=config,
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

        output = workflow.run(
            {
                const.S_CLASSIFICATION_INPUT: utterance,
                const.S_CONTEXT: context,
                const.S_INTENTS_INFO: intents_info,
                const.S_NER_INPUT: utterance,
                const.S_REFERENCE_TIME: reference_time,
                const.S_LOCALE: locale
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
