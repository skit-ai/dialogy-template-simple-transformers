import json

import pandas as pd
from dialogy.plugins.preprocess.text.calibration import (
    filter_asr_output,
)  # type: ignore
from dialogy.plugins.preprocess.text.normalize_utterance import (
    normalize,
)  # type: ignore
from tqdm import tqdm

from slu import constants as const  # type: ignore
from slu.dev.io.mp import parallel_proc  # type: ignore
from slu.dev.io.reader.csv import get_unique_labels  # type: ignore
from slu.dev.io.reader.csv import map_labels_in_df, read_multiclass_dataset_csv
from slu.dev.io.reader.sqlite import read_multiclass_dataset_sqlite  # type: ignore
from slu.utils.merge_configs import merge_calibration_config


def preprocess(df, calibration_config):
    calibration = merge_calibration_config(calibration_config)
    texts = []
    labels = []
    data_id = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        label = row[const.LABELS]
        data = json.loads(row[const.DATA])
        language = row[const.LANGUAGE]
        try:
            if language in calibration:
                alternatives = filter_asr_output(
                    data[const.ALTERNATIVES], **calibration[language]
                )[0]
            else:
                alternatives = data[const.ALTERNATIVES]

            data = normalize(alternatives)
            texts.append(data)
            labels.append(label)
            data_id.append(row[const.DATA_ID])
        except KeyError:
            raise KeyError(
                "Your data doesn't match the expected format!"
                ' your data column should have {"alternatives": [[{"transcript": "..."}]]})'
                f" \ninstead looks like {data} and each alternative should have am_score, lm_score"
            )
    return pd.DataFrame(
        {const.DATA_ID: data_id, const.TEXT: texts, const.LABELS: labels},
        columns=[const.DATA_ID, const.TEXT, const.LABELS],
    )


def read_multiclass_dataset(full_path, alias=None, file_format=const.CSV, **kwargs):
    if file_format == const.SQLITE:
        data_frame = read_multiclass_dataset_sqlite(full_path, **kwargs)
        return map_labels_in_df(data_frame, alias=alias)

    if file_format == const.CSV:
        return read_multiclass_dataset_csv(full_path, alias=alias, **kwargs)
    else:
        raise ValueError(
            "Expected format to be a string with"
            f" values in {const.V_SUPPORTED_DATA_FORMATS} but {file_format} was found."
        )


def prepare(
    data_file,
    alias,
    file_format=const.CSV,
    n_cores=const.N_DEFAULT_CORES,
    calibration_config=[],
):
    dataset = read_multiclass_dataset(
        data_file,
        alias,
        file_format=file_format,
        usecols=[const.DATA_ID, const.DATA, const.LABELS],
    )
    data_frame = parallel_proc(
        dataset,
        preprocess,
        calibration_config=calibration_config,
        return_df=True,
        n_cores=n_cores,
    )
    labels = get_unique_labels(data_frame, const.LABELS)
    return data_frame, labels
