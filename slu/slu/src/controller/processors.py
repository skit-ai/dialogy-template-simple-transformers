import os
from typing import Dict, List
from dialogy import plugins
from dialogy.base.plugin import Plugin

from slu.dev.plugin_parse import plugin_functions
from slu import constants as const
from slu.utils.config import Config


def get_preprocessors(config: Config) -> List[Plugin]:
    merge_asr_output = plugins.MergeASROutputPlugin(
        access=plugin_functions.access(const.INPUT, const.S_CLASSIFICATION_INPUT),
        mutate=plugin_functions.mutate(const.INPUT, const.S_CLASSIFICATION_INPUT),
    )()

    duckling_plugin = plugins.DucklingPlugin(
        access=plugin_functions.access(
            const.INPUT, const.S_NER_INPUT, const.S_REFERENCE_TIME, const.S_LOCALE
        ),
        mutate=plugin_functions.mutate(const.OUTPUT, const.ENTITIES),
        dimensions=["people", "number", "time", "duration"],
        locale="en_IN",
        timezone="Asia/Kolkata",
        timeout=0.5,
        # Works only in development mode. You need to set this in k8s configs.
        url=os.environ.get("DUCKLING_URL", "http://localhost:8000/parse/"),
    )()

    wer_calibration = plugins.WERCalibrationPlugin(config=config.calibration, access=None, mutate=None)()
    return [wer_calibration, merge_asr_output, duckling_plugin]


def get_postprocessors(config: Config) -> List[Plugin]:
    slot_filler = plugins.RuleBasedSlotFillerPlugin(
        access=plugin_functions.access(const.OUTPUT, const.INTENT, const.ENTITIES),
        rules=config.slots,
    )()
    return [slot_filler]
