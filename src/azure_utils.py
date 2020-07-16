from typing import Dict, Optional

import yaml
import os

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient


def create_container(
    container_name: str,
    blob_service_client: Optional[BlobServiceClient] = None,
    account_name: Optional[str] = None,
    account_key: Optional[str] = None,
    default_endpoints_protocol: str = "https",
    endpoint_suffix: str = "core.windows.net",
    exist_ok=True,
) -> BlobServiceClient:
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
    account_name: str,
    account_key: str,
    default_endpoints_protocol: str = "https",
    endpoint_suffix: str = "core.windows.net",
) -> str:
    return f"DefaultEndpointsProtocol={default_endpoints_protocol};AccountName={account_name};AccountKey={account_key};EndpointSuffix={endpoint_suffix}"


def load_azure_conf(conf_path: Optional[str] = None) -> Dict[str, str]:
    conf_path = conf_path or os.path.join("src", "azure_conf.yml")
    with open(conf_path, "r") as f:
        azure_conf = yaml.load(f, Loader=yaml.FullLoader)
    return azure_conf
