import uwsgi

from slu import constants as const
from slu.src.controller.prediction import get_predictions


class ThreadSafePredictAPI:
    def __init__(self, utterance, lang, config, context=dict, intents_info=None, history=list):
        self.utterance = utterance
        self.context = context
        self.intents_info = intents_info
        self.lang = lang
        self.history = history
        self.predict = get_predictions(const.PRODUCTION, config=config)

    def __enter__(self):
        uwsgi.lock()
        return self.predict(
            alternatives=self.utterance,
            context=self.context,
            intents_info=self.intents_info,
            history=self.history,
            lang=self.lang,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        uwsgi.unlock()
