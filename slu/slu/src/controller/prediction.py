"""
This module provides a simple interface to provide text features
and receive Intent and Entities.
"""
import importlib
from typing import Any, Dict, List, Optional
from dialogy.plugins.preprocess.text.normalize_utterance import normalize

from slu import constants as const
from slu.src.workflow import XLMRWorkflow
from slu.utils.config import Config
from slu.dev.plugin_parse.plugin_functional_arguments import plugin_param_parser


plugin_module = importlib.import_module("dialogy.plugins")


def predict_wrapper(config_map: Dict[str, Config]):
    """
    Create a closure for the predict function.

    Ensures that the workflow is loaded just once without creating global variables for it.
    This can also be made into a class if needed.
    """
    config: Config = list(config_map.values()).pop()

    preprocessors = []
    for plugin_config in config.preprocess:
        plugin_name = plugin_config[const.PLUGIN]
        plugin_params = {key: plugin_param_parser(value) for key, value in plugin_config[const.PARAMS].items()}
        plugin_container = getattr(plugin_module, plugin_name)
        plugin = plugin_container(**plugin_params)
        preprocessors.append(plugin())

    postprocessors = []
    for plugin_config in config.postprocess:
        plugin_name = plugin_config[const.PLUGIN]
        plugin_params = plugin_config[const.PARAMS]
        plugin_container = getattr(plugin_module, plugin_name)
        plugin = plugin_container(**plugin_params)
        postprocessors.append(plugin())

    workflow = XLMRWorkflow(
        preprocessors=preprocessors,
        postprocessors=postprocessors,
        config=config
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
