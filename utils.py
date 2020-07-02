import numpy as np
from tensorflow.keras.utils import to_categorical
import yaml


def load_training_conf():
    with open("train_conf.yml", "r") as file:
        conf = yaml.full_load(file)
    return conf


def encode_texts(tokenizer, texts):
    return np.array(
        [
            tokenizer.encode(
                text,
                max_length=tokenizer.max_length,
                pad_to_max_length=tokenizer.pad_to_max_length,
            )
            for text in texts
        ]
    )


def encode_labels(texts_labels, unique_labels):
    unique_labels = sorted(unique_labels)
    label_int = dict((label, i) for (i, label) in enumerate(unique_labels))
    # int_label = dict((i, label) for (i, label) in enumerate(unique_labels))

    return to_categorical(
        np.array([label_int[text_label] for text_label in texts_labels])
    )
