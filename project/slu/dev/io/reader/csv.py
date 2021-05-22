import os
from tqdm import tqdm  # type: ignore
import pandas as pd  # type: ignore

from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as multi_label_report

from slu import constants as const


def map_label(label, alias):
    """
    Update label from its alias value in config.yaml.

    ```
    alias.get(label, label)
    ```
    If the `label` has an alias, use it else use the label as-is.
    This requires labels to be created this way:

    ```yaml
    alias:
        aliased_label_1: real_label
        aliased_label_2: real_label
    ```
    as opposed to

    ```yaml
    alias:
        real_label:
            - aliased_label_1
            - aliased_label_2
    ```
    The above snippet requires an algorithm of O^3 and more lines of code than present here.
    """
    if not alias:
        return label
    return alias.get(label, label)


def get_unique_labels(data_frame, label_column):
    """
    return unique items in a `pd.DataFrame` column.
    """
    return data_frame[label_column].unique().tolist()


def map_labels_in_df(data_frame, alias=None):
    """
    Map label fields with their alias' for each row in DataFrame.
    """
    if not alias:
        return data_frame

    for i, row in tqdm(data_frame.iterrows(), total=len(data_frame)):
        label = map_label(row[const.LABELS], alias)
        data_frame.loc[i, const.LABELS] = label

    return data_frame


def read_multiclass_dataset_csv(full_path, alias=None, **kwargs):
    """
    Read a dataset that supports `simpletransformers.classification` classes.
    """
    data_frame = pd.read_csv(full_path, **kwargs)
    return map_labels_in_df(data_frame, alias=alias)


def read_ner_dataset_csv(full_path, **kwargs):
    """
    Read dataset that supports `simpletransformers.ner` classes.
    """
    data_frame = pd.read_csv(full_path, **kwargs)[
        [const.SENTENCE_ID, const.WORDS, const.LABELS]
    ]
    labels = get_unique_labels(data_frame, const.LABELS)
    return data_frame, labels


def save_report(classification_report_output, metrics_dir):
    """
    Save f1-measure metrics as csv.
    """
    report = pd.DataFrame(classification_report_output).T
    report.index.names = [const.METRICS]
    report.to_csv(os.path.join(metrics_dir, const.S_REPORT))


def save_classification_report(true_labels, pred_labels, metrics_dir):
    """
    Save f1-measure reports for classification tasks.
    """
    report = classification_report(true_labels, pred_labels, digits=2, output_dict=True)
    save_report(report, metrics_dir)


def save_ner_report(results, metrics_dir):
    """
    Save f1-measure reports for ner tasks.
    """
    save_report(results[const.REPORT], metrics_dir)
