import yaml
import os

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient


def create_container(
    container_name,
    blob_service_client=None,
    account_name=None,
    account_key=None,
    default_endpoints_protocol="https",
    endpoint_suffix="core.windows.net",
    exist_ok=True,
):
    if blob_service_client is None:
        assert account_name is not None and account_key is not None
        blob_service_client = BlobServiceClient.from_connection_string(
            blob_connection_string(
                account_name,
                account_key,
                default_endpoints_protocol=default_endpoints_protocol,
                endpoint_suffix=endpoint_suffix,
            )
        )
    try:
        blob_service_client.create_container(container_name)
    except ResourceExistsError:
        if exist_ok:
            pass
        else:
            raise ResourceExistsError
    return blob_service_client


def blob_connection_string(
    account_name,
    account_key,
    default_endpoints_protocol="https",
    endpoint_suffix="core.windows.net",
):
    return f"DefaultEndpointsProtocol={default_endpoints_protocol};AccountName={account_name};AccountKey={account_key};EndpointSuffix={endpoint_suffix}"


def load_azure_conf(conf_path=None):
    conf_path = conf_path or os.path.join("src", "azure_conf.yml")
    with open(conf_path, "r") as f:
        azure_conf = yaml.load(f, Loader=yaml.FullLoader)
    return azure_conf
