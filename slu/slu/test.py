import pandas as pd

from slu.dev.prepare import preprocess
from slu.utils.config import Config, YAMLLocalConfig

CLIENT_CONFIGS = YAMLLocalConfig().generate()
config = list(CLIENT_CONFIGS.values()).pop()

df = pd.read_csv("train-en-utterances.csv")
df["labels"] = "confirm"
df["data"] = df["alternative-data"]
prepare(df, config.calibration)
