"""
This module provides a simple interface to provide text features
and receive Intent and Entities.
"""
import importlib
import traceback
from typing import Any, Dict, List, Optional
from dialogy.plugins.preprocess.text.normalize_utterance import normalize

from slu import constants as const
from slu.src.workflow import XLMRWorkflow
from slu.utils.config import Config
from slu.dev.plugin_parse.plugin_functional_arguments import plugin_param_parser
from slu.utils.logger import log


plugin_module = importlib.import_module("dialogy.plugins")


def parse_plugin_params(plugins):
    plugin_list = []
    for plugin_config in plugins:
        plugin_name = plugin_config[const.PLUGIN]
        plugin_params = {key: plugin_param_parser(value) for key, value in plugin_config[const.PARAMS].items()}
        plugin_container = getattr(plugin_module, plugin_name)
        try:
            plugin = plugin_container(**plugin_params)
            plugin_list.append(plugin())
        except (TypeError, ValueError) as error:
            log.error("Seems like the slot definitions are missing or incorrect."
            f" To setup {plugin_name} you need to provide the params via entity definitions in the slots."
            "If this message is not clear by itself, refer to https://gist.github.com/greed2411/be114ba10e29196a995af8423c98399b for a template." 
            f"{error}")
            log.error(traceback.format_exc())
            log.error(f"{plugin_name} was not added to the list of plugins, your workflow will operate but without {plugin_name}.")
    return plugin_list


def predict_wrapper(config_map: Dict[str, Config]):
    """
    Create a closure for the predict function.

    Ensures that the workflow is loaded just once without creating global variables for it.
    This can also be made into a class if needed.
    """
    config: Config = list(config_map.values()).pop()

    preprocessors = parse_plugin_params(config.preprocess)
    postprocessors = parse_plugin_params(config.postprocess)

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
