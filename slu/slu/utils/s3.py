import io
from typing import Set, Union

import boto3
import pandas as pd
import requests
from pandas.core.algorithms import isin


def get_private_csv_from_s3(file_key: str, bucket: str):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=file_key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def get_public_csv(url: str, *args):
    content = requests.get(url).content
    return pd.read_csv(io.StringIO(content.decode("utf-8")))


def get_csvs(urls: Union[str, Set[str]], bucket: str = None, fn=get_public_csv):
    if isinstance(urls, str):
        url = urls
        return fn(url, bucket)
    elif isinstance(urls, set):
        return pd.concat([fn(url, bucket) for url in urls if isinstance(url, str)])
    else:
        raise TypeError(
            f"file_key should be a Set[str] or str but {urls}<{type(urls)}> was found."
        )
