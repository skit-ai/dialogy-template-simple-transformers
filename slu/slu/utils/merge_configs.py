#   Merge calibration config into
#   a single dict
#

import pickle
from functools import reduce
from operator import add


def merge_calibration_config(calibration_config):
    calibration = {}
    for lang in calibration_config:
        calibration[lang] = dict(
            reduce(add, map(list, (map(dict.items, calibration_config[lang]))))
        )
        calibration[lang]["vectorizer"] = pickle.load(
            open(calibration[lang]["vectorizer_path"], "rb")
        )
        calibration[lang]["classifier"] = pickle.load(
            open(calibration[lang]["classifier_path"], "rb")
        )
        calibration[lang].pop("vectorizer_path")
        calibration[lang].pop("classifier_path")
    return calibration
