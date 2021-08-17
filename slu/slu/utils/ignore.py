from slu import constants as const


def ignore_utterance(utterance):
    return all(uttr in const.TEXTS_TO_IGNORE for uttr in utterance)
