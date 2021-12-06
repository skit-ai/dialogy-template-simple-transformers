import os
from typing import List

from dialogy import plugins
from dialogy.base.plugin import Plugin
from dialogy.workflow import Workflow

from slu import constants as const
from slu.dev.plugin_parse import plugin_functions
from slu.src.controller.custom_plugins import ContextualIntentSwap
from slu.utils.config import Config


def get_plugins(purpose, config: Config, debug=False) -> List[Plugin]:
    duckling_plugin = plugins.DucklingPlugin(
        access=plugin_functions.duckling_access,
        mutate=plugin_functions.mutate(const.OUTPUT, const.ENTITIES),
        dimensions=["people", "number", "time", "duration"],
        locale="en_IN",
        timezone="Asia/Kolkata",
        timeout=0.5,
        input_column=const.ALTERNATIVES,
        output_column=const.ENTITIES,
        # url works only in development mode.
        # You need to set its real value in k8s configs or wherever you keep your
        # env-vars safe.
        url=os.environ.get("DUCKLING_URL", "http://localhost:8000/parse/"),
        use_transform=True,
        debug=debug,
    )

    list_entity_plugin = plugins.ListEntityPlugin(
        access=plugin_functions.access(const.INPUT, const.NER_INPUT),
        mutate=plugin_functions.mutate(const.OUTPUT, const.ENTITIES),
        style=const.REGEX,
        candidates=config.entity_patterns,
        threshold=0.1,
        input_column=const.ALTERNATIVES,
        output_column=const.ENTITIES,
        use_transform=True,
        debug=debug,
    )

    canonicalizer = plugins.CanonicalizationPlugin(
        input_column=const.ALTERNATIVES,
        entity_column=const.ENTITIES,
        use_transform=True,
        access=lambda w: (w.output[const.ENTITIES], w.input[const.NER_INPUT]),
        mutate=plugin_functions.mutate(
            const.INPUT, const.CLASSIFICATION_INPUT, action=const.REPLACE
        )
    )

    merge_asr_output = plugins.MergeASROutputPlugin(
        access=plugin_functions.access(const.INPUT, const.CLASSIFICATION_INPUT),
        mutate=plugin_functions.mutate(
            const.INPUT, const.CLASSIFICATION_INPUT, action=const.REPLACE
        ),
        use_transform=True,
        input_column=const.ALTERNATIVES,
        debug=debug,
    )

    xlmr_clf = plugins.XLMRMultiClass(
        model_dir=config.get_model_dir(const.CLASSIFICATION),
        access=plugin_functions.access(const.INPUT, const.CLASSIFICATION_INPUT),
        mutate=plugin_functions.mutate(const.OUTPUT, const.INTENTS),
        threshold=config.get_model_confidence_threshold(const.CLASSIFICATION),
        score_round_off=5,
        purpose=purpose,
        use_cuda=purpose != const.PRODUCTION,
        data_column=const.ALTERNATIVES,
        label_column=const.TAG,
        args_map=config.get_model_args(const.CLASSIFICATION),
        debug=debug,
    )

    contextual_name_swapper = ContextualIntentSwap(
        access=lambda w: (w.output[const.INTENTS], w.input[const.CONTEXT], w.output[const.ENTITIES]),
    )

    slot_filler = plugins.RuleBasedSlotFillerPlugin(
        access=plugin_functions.access(const.OUTPUT, const.INTENTS, const.ENTITIES),
        rules=config.slots,
        debug=debug,
        fill_multiple=True,
    )

    return [merge_asr_output, xlmr_clf, slot_filler]
