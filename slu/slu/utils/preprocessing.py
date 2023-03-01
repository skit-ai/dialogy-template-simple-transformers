import json
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from slu import constants as const
from slu.utils import logger
from slu.utils.config import YamlAliasConfig

def make_label_column_uniform(data_frame: pd.DataFrame, alias_yaml:str) -> pd.DataFrame:
    alias_map = YamlAliasConfig(config_path=alias_yaml).generate()

    if const.INTENTS in data_frame.columns:
        column = const.INTENTS
    elif const.LABELS in data_frame.columns:
        column = const.LABELS
    elif const.TAG in data_frame.columns:
        column = const.TAG
    else:
        raise ValueError(
            f"Expected one of {const.LABELS}, {const.TAG} to be present in the dataset."
        )
    data_frame.rename(columns={column: const.TAG}, inplace=True)
    data_frame = data_frame.replace({const.TAG: alias_map})
    return data_frame

def reftime_patterns(reftime: str):
    time_fns = [
        datetime.fromisoformat,
        lambda date_string: datetime.strptime(
            date_string, "%Y-%m-%d %H:%M:%S.%f %z %Z"
        ),
        lambda date_string: datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ"),
        lambda date_string: datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S.%f%z"),
    ]
    for time_fn in time_fns:
        try:
            return time_fn(reftime)
        except ValueError:
            continue
    raise ValueError(f"Could not parse reftime {reftime}")


def make_reftime_column_uniform(data_frame: pd.DataFrame) -> pd.DataFrame:
    if const.REFERENCE_TIME not in data_frame.columns:
        return

    for i, row in tqdm(
        data_frame.iterrows(), total=len(data_frame), desc="Fixing reference time"
    ):
        if row[const.REFERENCE_TIME] is not None and not pd.isna(
            row[const.REFERENCE_TIME]
        ):
            data_frame.loc[i, const.REFERENCE_TIME] = reftime_patterns(
                row[const.REFERENCE_TIME]
            ).isoformat()

    return data_frame

def make_data_column_uniform(data_frame: pd.DataFrame) -> pd.DataFrame:
    if const.ALTERNATIVES in data_frame.columns:
        column = const.ALTERNATIVES
    elif const.DATA in data_frame.columns:
        column = const.DATA
    else:
        raise ValueError(
            f"Expected one of {const.ALTERNATIVES}, {const.DATA} to be present in the dataset."
        )
    data_frame.rename(columns={column: const.ALTERNATIVES}, inplace=True)

    for i, row in tqdm(
        data_frame.iterrows(), total=len(data_frame), desc="Fixing data structure"
    ):
        if isinstance(row[const.ALTERNATIVES], str):
            data = json.loads(row[const.ALTERNATIVES])
            if isinstance(data, str):
                data = json.loads(row[const.ALTERNATIVES])
            if const.ALTERNATIVES in data:
                data_frame.loc[i, const.ALTERNATIVES] = json.dumps(
                    data[const.ALTERNATIVES], ensure_ascii=False
                )
                
    return data_frame
