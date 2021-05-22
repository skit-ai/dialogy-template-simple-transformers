import sqlite3

import pandas as pd
from tqdm import tqdm

DEFAULT_QUERY = "SELECT {cols} FROM data"
DEFAULT_COLS = ["*"]


def read_sqlite(db, query=DEFAULT_QUERY, usecols=None):
    if not usecols:
        usecols = DEFAULT_COLS

    query = query.format(cols=", ".join(usecols))
    connection = sqlite3.connect(db)
    cursor = connection.cursor()
    cursor.execute(query)
    columns = [description[0] for description in cursor.description]
    return columns, cursor.fetchall()


def read_multiclass_dataset_sqlite(full_path, alias=None, **kwargs):
    columns, data = read_sqlite(full_path, query=DEFAULT_QUERY, **kwargs)
    return pd.DataFrame(data, columns=columns)
