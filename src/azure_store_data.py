import yaml
import os
from azure.core.exceptions import ResourceExistsError

from src.azure_utils import create_container
from azure.storage.blob import BlobServiceClient


def write_csv_azure(
    csv: str,
    csv_name: str,
    blob_service_client: BlobServiceClient,
    container_name: str,
    exist_ok: bool = True,
) -> None:
    try:
        blob_service_client.get_blob_client(container_name, csv_name).upload_blob(csv)
    except ResourceExistsError:
        if exist_ok:
            pass
        else:
            raise ResourceExistsError


def load_csv_as_str(csv_path: str) -> str:
    with open(csv_path) as f:
        csv = "".join(f.readlines())
    return csv


if __name__ == "__main__":

    with open(os.path.join("azure", "azure_conf.yml")) as f:
        azure_conf = yaml.load(f, Loader=yaml.FullLoader)

    blob_service_client = create_container(
        azure_conf["CONTAINER_NAME"],
        account_name=azure_conf["STORAGE"]["AccountName"],
        account_key=azure_conf["STORAGE"]["AccountKey"],
        exist_ok=True,
    )
    write_csv_azure(
        csv=load_csv_as_str(azure_conf["LOCAL_DATASET_PATH"]),
        csv_name=azure_conf["LOCAL_DATASET_PATH"].split(os.sep)[-1],
        blob_service_client=blob_service_client,
        container_name=azure_conf["CONTAINER_NAME"],
    )
    print("Data uploaded to Azure!")
