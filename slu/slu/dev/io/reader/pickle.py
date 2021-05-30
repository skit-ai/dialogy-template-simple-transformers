import os
import pickle

from slu import constants as const


def save_label_encoder(model_dir, encoder):
    with open(os.path.join(model_dir, const.S_INTNET_LABEL_ENCODER), "wb") as handle:
        pickle.dump(encoder, handle)


def read_label_encoder(model_dir):
    with open(os.path.join(model_dir, const.S_INTNET_LABEL_ENCODER), "rb") as handle:
        return pickle.load(handle)


def save_intent_labels(model_dir, labels):
    with open(os.path.join(model_dir, const.S_ENTITY_LABELS), "wb") as f:
        pickle.dump(labels, f)


def read_intent_labels(model_dir):
    with open(os.path.join(model_dir, const.S_ENTITY_LABELS), "rb") as f:
        pickle.load(f)
