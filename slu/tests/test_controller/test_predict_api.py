from datetime import datetime
from slu.src.controller.prediction import predict_wrapper
from slu.utils.config import YAMLLocalConfig
from slu import constants as const


CLIENT_CONFIGS = YAMLLocalConfig().generate()
PREDICT_API = predict_wrapper(CLIENT_CONFIGS)


def test_no_input_yields_fallback():
    """Test predict_api with no input."""
    maybe_utterance = []
    context = {}
    intents_info = []
    lang = "en"

    response = PREDICT_API(
                maybe_utterance,
                context,
                intents_info=intents_info,
                reference_time=int(datetime.now().timestamp() * 1000),
                locale=const.LANG_TO_LOCALES[lang],
                lang=lang,
            )
    assert response[const.INTENTS][0][const.NAME] == const.S_INTENT_OOS


def test_utterances():
    """Test predict_api with no input."""
    maybe_utterance = [[{"transcript": "hello"}]]
    context = {}
    intents_info = []
    lang = "en"

    response = PREDICT_API(
                maybe_utterance,
                context,
                intents_info=intents_info,
                reference_time=int(datetime.now().timestamp() * 1000),
                locale=const.LANG_TO_LOCALES[lang],
                lang=lang,
            )
    assert response[const.INTENTS][0][const.NAME] == const.S_INTENT_OOS
