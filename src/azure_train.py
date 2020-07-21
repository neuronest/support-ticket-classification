import os
import argparse
import shutil
from typing import Optional, Tuple

from tensorflow.keras.callbacks import Callback
from azureml.core import Run

from utils import load_training_conf
from model import DistilBertClassifier, save_model
from train import training_data, train_model, define_callbacks
from azure_utils import load_azure_conf


class LogRunMetrics(Callback):
    """
    Azure ML metrics logger, a run context is used globally
    """

    def on_epoch_end(self, epoch: int, log: Optional[dict] = None) -> None:
        if "val_loss" in log and "val_accuracy" in log:
            azure_run_context.log("Loss", log["val_loss"])
            azure_run_context.log("Accuracy", log["val_accuracy"])


def handle_arguments(arg_parser) -> argparse.Namespace:
    # data args
    arg_parser.add_argument("--data-folder", type=str)
    args = arg_parser.parse_args()
    return args


def handle_configurations() -> Tuple[dict, dict, dict]:
    conf = load_training_conf("train_conf.yml")
    conf_train, conf_data = conf["training"], conf["data"]
    azure_conf = load_azure_conf("azure_conf.yml")
    return conf_train, conf_data, azure_conf


if __name__ == "__main__":
    azure_run_context = Run.get_context()
    args = handle_arguments(argparse.ArgumentParser())
    conf_train, conf_data, azure_conf = handle_configurations()
    csv_dataset_name = azure_conf["LOCAL_DATASET_PATH"].split(os.sep)[-1]

    (x_train, x_test, y_train, y_test), tokenizer = training_data(
        tickets_data_path=os.path.join(args.data_folder, csv_dataset_name),
        text_column=conf_data["text_column"],
        label_column=conf_data["label_column"],
        test_size=conf_train.get("test_set_size", 0.25),
        subset_size=-1,
        max_length=conf_data["max_words_per_message"],
        pad_to_max_length=conf_data.get("pad_to_max_length", True),
    )
    model = DistilBertClassifier(
        num_labels=y_train.shape[1],
        learning_rate=conf_train.get("learning_rate", 5e-5),
    )
    callbacks = define_callbacks(
        patience=conf_train.get("early_stopping_patience", 3),
        min_delta=conf_train.get("early_stopping_min_delta_acc", 0.01),
    ) + [LogRunMetrics()]
    train_model(
        model,
        x_train,
        x_test,
        y_train,
        y_test,
        epochs=conf_train.get("epochs", 1),
        batch_size=conf_train.get("batch_size", 64),
        callbacks=callbacks,
    )
    save_model(model, tokenizer, "outputs")
    shutil.make_archive(
        os.path.join("outputs", "my_model"),
        "gztar",
        os.path.join("outputs", "my_model"),
    )
