import os
from typing import List, Any, Union, Tuple, Sequence

from transformers import TFDistilBertForSequenceClassification
from transformers.tokenization_distilbert import DistilBertTokenizer
import tensorflow as tf
import pickle

# from src.utils import encode_texts
from utils import encode_texts


class DistilBertClassifier(tf.keras.Model):
    def __init__(
        self,
        num_labels: int,
        learning_rate: float = 5e-5,
        dropout_rate: float = 0.2,
        metrics: List[str] = ["accuracy"],
    ):
        super(DistilBertClassifier, self).__init__()
        hugging_face_distil_classifier = TFDistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-cased"
        )
        distil_classifier_out_dim = hugging_face_distil_classifier.config.dim
        self.distilbert = hugging_face_distil_classifier.get_layer("distilbert")

        self.dense1 = tf.keras.layers.Dense(
            distil_classifier_out_dim, activation="relu", name="dense1"
        )
        self.dense2 = tf.keras.layers.Dense(
            distil_classifier_out_dim // 2, activation="relu", name="dense2"
        )
        self.dense3 = tf.keras.layers.Dense(
            distil_classifier_out_dim // 4, activation="relu", name="dense3"
        )
        self.dense4 = tf.keras.layers.Dense(num_labels, name="dense4")
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.compile(
            loss=loss_fn, optimizer=tf.optimizers.Adam(learning_rate), metrics=metrics
        )

    def call(
        self, inputs: tf.Tensor, **kwargs: Any
    ) -> Tuple[tf.Tensor, Union[tf.Tensor, None]]:
        distilbert_output = self.distilbert(inputs, **kwargs)

        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.dense1(pooled_output)
        pooled_output = self.dense2(pooled_output)
        pooled_output = self.dense3(pooled_output)
        logits = self.dense4(pooled_output)  # (bs, dim)

        outputs = (logits,) + distilbert_output[1:]
        return outputs


def save_model(
    model: DistilBertClassifier, tokenizer: DistilBertTokenizer, model_folder="."
) -> None:
    os.makedirs(model_folder, exist_ok=True)
    model.save(os.path.join(model_folder, "my_model"))
    with open(os.path.join(model_folder, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)


def load_model(
    model_folder: str = ".",
) -> Tuple[DistilBertClassifier, DistilBertTokenizer]:
    model = tf.keras.models.load_model(os.path.join(model_folder, "my_model"))
    with open(os.path.join(model_folder, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


def model_predict(
    model: DistilBertClassifier, tokenizer: DistilBertTokenizer, texts: Sequence[str]
):
    return model.predict(encode_texts(tokenizer, texts)).argmax(axis=1)
