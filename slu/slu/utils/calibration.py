#   Merge calibration config into
#   a single dict
#
import os
import pickle


def safe_load(file_path, load=None):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return load(f)
    else:
        return None


def prepare_calibration_config(calibration_config):
    calibration = {}
    for lang in calibration_config:
        calibration[lang] = {}
        vectorizer_path = calibration_config[lang]["vectorizer_path"]
        classifier_path = calibration_config[lang]["classifier_path"]

        calibration[lang]["vectorizer"] = safe_load(vectorizer_path, pickle.load)
        calibration[lang]["classifier"] = safe_load(classifier_path, pickle.load)
    return calibration
