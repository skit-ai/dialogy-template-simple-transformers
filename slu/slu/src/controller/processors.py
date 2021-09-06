import os
from typing import List

from dialogy import plugins
from dialogy.base.plugin import Plugin

from slu import constants as const
from slu.dev.plugin_parse import plugin_functions
from slu.utils.config import Config


def get_plugins(purpose, config: Config, debug=False) -> List[Plugin]:
    merge_asr_output = plugins.MergeASROutputPlugin(
        access=plugin_functions.access(const.INPUT, const.CLASSIFICATION_INPUT),
        mutate=plugin_functions.mutate(
            const.INPUT, const.CLASSIFICATION_INPUT, action=const.REPLACE
        ),
        data_column=const.ALTERNATIVES,
        debug=debug,
    )

    duckling_plugin = plugins.DucklingPlugin(
        access=plugin_functions.access(
            const.INPUT, const.NER_INPUT, const.REFERENCE_TIME, const.LOCALE
        ),
        mutate=plugin_functions.mutate(const.OUTPUT, const.ENTITIES),
        dimensions=["people", "number", "time", "duration"],
        locale="en_IN",
        timezone="Asia/Kolkata",
        timeout=0.5,
        # url works only in development mode.
        # You need to set its real value in k8s configs or wherever you keep your
        # env-vars safe.
        url=os.environ.get("DUCKLING_URL", "http://localhost:8000/parse/"),
        debug=debug,
    )

    xlmr_clf = plugins.XLMRMultiClass(
        model_dir=config.get_model_dir(const.CLASSIFICATION),
        access=plugin_functions.access(const.INPUT, const.CLASSIFICATION_INPUT),
        mutate=plugin_functions.mutate(const.OUTPUT, const.INTENTS),
        threshold=config.get_model_confidence_threshold(const.CLASSIFICATION),
        score_round_off=5,
        purpose=purpose,
        use_cuda=purpose == const.PRODUCTION,
        data_column=const.ALTERNATIVES,
        label_column=const.INTENT,
        args_map=config.get_model_args(const.CLASSIFICATION),
        debug=debug,
    )

    slot_filler = plugins.RuleBasedSlotFillerPlugin(
        access=plugin_functions.access(const.OUTPUT, const.INTENTS, const.ENTITIES),
        rules=config.slots,
        debug=debug,
    )

    return [merge_asr_output, duckling_plugin, xlmr_clf, slot_filler]
