"""
Testing routine.

Usage:
  test.py <version>
  test.py (classification|ner) <version>
  test.py (-h | --help)
  test.py --version

Options:
    <version>   The version of the dataset to use, the model produced will also be in the same dir.
    -h --help   Show this screen.
    --version   Show version.
"""
from functools import partial

import numpy as np
import pandas as pd
from docopt import docopt
from seqeval.metrics import classification_report as multi_label_report
from sklearn.metrics import accuracy_score, classification_report

from slu import constants as const
from slu.utils.config import Config
from slu.utils.logger import log


def test_classifier(config, file_format=const.CSV, custom_file=None):
    log.info("Preparing dataset.")
    test_data_frame = config.get_dataset(
        const.CLASSIFICATION,
        const.TEST,
        file_format=file_format,
        custom_file=custom_file,
    )
    labelencoder = config.load_pickle(
        const.CLASSIFICATION, const.TEST, const.S_INTENT_LABEL_ENCODER
    )
    test_data_frame[const.LABELS] = labelencoder.transform(
        test_data_frame[const.LABELS]
    )
    model = config.get_model(const.CLASSIFICATION, const.TEST)

    _, model_outputs, _ = model.eval_model(
        test_data_frame,
        acc=accuracy_score,
        report=classification_report,
    )

    pred_labels = labelencoder.inverse_transform(np.argmax(model_outputs, axis=1))
    true_labels = labelencoder.inverse_transform(test_data_frame[const.LABELS])

    log.info("saving report.")
    config.save_report(const.CLASSIFICATION, (true_labels, pred_labels))

    errors = []
    for i, (pred_label, true_label) in enumerate(zip(pred_labels, true_labels)):
        if pred_label != true_label:
            errors.append(
                {
                    const.DATA_ID: test_data_frame[const.DATA_ID].iloc[i],
                    const.DATA: test_data_frame[const.TEXT].iloc[i],
                    const.TRUE_LABEL: true_label,
                    const.PRED_LABEL: pred_label,
                }
            )
    data_frame = pd.DataFrame(
        errors, columns=[const.DATA_ID, const.DATA, const.TRUE_LABEL, const.PRED_LABEL]
    )
    config.save_classification_errors(data_frame)


def test_ner(config, file_format=const.CSV, custom_file=None):
    log.info("Preparing dataset.")
    test_data_frame = config.get_dataset(
        const.NER, const.TEST, file_format=file_format, custom_file=custom_file
    )
    report_fn = partial(multi_label_report, output_dict=True)
    model = config.get_model(const.NER, const.TEST)

    results, _, _ = model.eval_model(test_data_frame, report=report_fn, verbose=False)

    log.info("saving report.")
    config.save_report(const.NER, results)
