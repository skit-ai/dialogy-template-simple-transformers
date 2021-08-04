"""
This module provides a simple interface to provide text features
and receive Intent and Entities.
"""
import os
import pickle
from functools import reduce
from operator import add
from typing import Any, Dict, List, Optional

from dialogy import plugins
from dialogy.plugins.preprocess.text.calibration import filter_asr_output
from dialogy.plugins.preprocess.text.normalize_utterance import normalize

from slu import constants as const
from slu.dev.plugin_parse import plugin_functions
from slu.src.workflow import XLMRWorkflow
from slu.utils.config import Config
from slu.utils.logger import log
from slu.utils.ignore import ignore_utterance

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
    calibration = merge_calibration_config(config.calibration)

    def predict(
        utterance: List[str],
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
        use_calibration = lang in calibration
        if use_calibration:
            filtered_utterance, predicted_wers = filter_asr_output(
                utterance, **calibration["lang"]
            )
            if len(predicted_wers) == 0:
                return {
                    const.VERSION: config.VERSION,
                    const.INTENTS: [{"name": "_oos_"}],
                    const.ENTITIES: [],
                }
            filtered_utterance = normalize(filtered_utterance)
            filtered_utterance_lengths = [
                len(uttr.split()) for uttr in filtered_utterance
            ]

            output_calibration = workflow.run(
                {
                    const.S_CLASSIFICATION_INPUT: filtered_utterance,
                    const.S_CONTEXT: context,
                    const.S_INTENTS_INFO: intents_info,
                    const.S_NER_INPUT: [],
                    const.S_REFERENCE_TIME: reference_time,
                    const.S_LOCALE: locale,
                }
            )
            intent_calibration = output_calibration[const.INTENT]
            if sum(filtered_utterance_lengths) / len(filtered_utterance_lengths) > 2:
                if sum(predicted_wers) / len(prediced_wers) > 0.9:
                    intent_calibration = [{"name": "_oos_"}]

        utterance = normalize(utterance)

        if ignore_utterance(utterance):
            return {
                const.VERSION: config.VERSION,
                const.INTENTS: [{"name": "_oos_"}],
                const.ENTITIES: [],
            }

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
        intent = output[const.INTENT] if not use_calibration else intent_calibration
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
