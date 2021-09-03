import pytest

from datetime import datetime
from slu.src.controller.prediction import get_predictions
from slu.utils.config import Config, YAMLLocalConfig
from slu import constants as const
from tests import load_tests


CONFIG_MAP = YAMLLocalConfig().generate()
CONFIG: Config = list(CONFIG_MAP.values()).pop()
PREDICT_API = get_predictions(const.PRODUCTION, config=CONFIG)


@pytest.mark.parametrize("payload", load_tests("cases", __file__))
def test_utterances(payload):
    """Test predict_api with inputs."""
    input_ = payload["input"]
    expected = payload["output"]
    output = PREDICT_API(**input_)
    pred_intent_name = output[const.INTENTS][0][const.NAME]
    true_intent_name = expected[const.INTENTS][0][const.NAME]
    pred_slots = {
        slot[const.NAME]: slot for slot in output[const.INTENTS][0][const.SLOTS]
    }
    true_slots = {
        slot[const.NAME]: slot for slot in expected[const.INTENTS][0][const.SLOTS]
    }

    assert pred_intent_name == true_intent_name, "intent name doesn't match!"
    assert len(pred_slots) == len(true_slots), "slot-size doesn't match!"
    for pred_slot, true_slot in zip(pred_slots, true_slots):
        pred_slot_values = pred_slots[pred_slot][const.VALUES]
        true_slot_values = true_slots[true_slot][const.VALUES]
        assert len(pred_slot_values) == len(
            true_slot_values
        ), f"Values for slot={true_slot} doesn't match!"
        for pred_slot_value, true_slot_value in zip(pred_slot_values, true_slot_values):
            assert (
                pred_slot_value[const.TYPE] == true_slot_value[const.TYPE]
            ), f"Type for slot={true_slot_value[const.TYPE]} doesn't match!"
            assert (
                pred_slot_value[const.VALUE] == true_slot_value[const.VALUE]
            ), f"Values for slot={true_slot_value[const.TYPE]} doesn't match!"
