import sqlite3
import pandas as pd
from tqdm import tqdm


DEFAULT_QUERY = "SELECT {cols} FROM data"
DEFAULT_COLS = ["*"]


def pre_process_client(data):
    """
    this functions extract data and label from the tog-cli sqlites in a format that can be directly used with dialogy train
    edit this function if you want to extract columns other than data and labels
    """

    full_data=[]
    for i in data:
        text = json.loads(i[1])['alternatives']
        text_dict={"alternatives":text}
        text_dict = json.dumps(text_dict)
        label = json.loads(i[2])[0]['type']
        data_tuple=(text_dict,label) 
        full_data.append(data_tuple)
    return full_data



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
    _, data = read_sqlite(full_path, query=DEFAULT_QUERY, **kwargs)
    data = pre_process_client(data)
    columns=['data','labels'] # edit this if you want more columns
    return pd.DataFrame(data, columns=columns)
