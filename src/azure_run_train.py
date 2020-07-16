"""
Endpoint to launch an experiment on AzureML.
"""

import os
from os.path import dirname
from typing import Optional

from azureml.train.estimator import Estimator
from azureml.core import Workspace, Datastore, Experiment, Run

from src.utils import pip_packages
from src.azure_utils import load_azure_conf


def run_azure_experiment_with_storage(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    datastore_name: str,
    container_name: str,
    storage_account_name: str,
    storage_account_key: str,
    compute_name: str,
    experiment_name: Optional[str] = None,
    source_directory: Optional[str] = None,
    image_name: Optional[str] = None,
    use_gpu=True,
) -> Run:
    workspace = Workspace(subscription_id, resource_group, workspace_name,)
    data_store = Datastore.register_azure_blob_container(
        workspace=workspace,
        datastore_name=datastore_name,
        container_name=container_name,
        account_name=storage_account_name,
        account_key=storage_account_key,
    )
    source_directory = source_directory or dirname(__file__)
    assert (
        compute_name in workspace.compute_targets
    ), f"compute {compute_name} is not created in {workspace_name} workspace"
    estimator = Estimator(
        source_directory=source_directory,
        script_params={"--data-folder": data_store.as_mount()},
        compute_target=workspace.compute_targets[compute_name],
        pip_packages=pip_packages(),
        entry_script=os.path.join(source_directory, "azure_train.py"),
        use_gpu=use_gpu,
        custom_docker_image=image_name,
    )
    experiment_name = experiment_name or __file__.split(os.sep)[-1].split(".py")[0]
    experiment = Experiment(workspace=workspace, name=experiment_name)
    run = experiment.submit(estimator)
    return run


if __name__ == "__main__":
    azure_conf = load_azure_conf()
    run = run_azure_experiment_with_storage(
        subscription_id=azure_conf["SUBSCRIPTION_ID"],
        resource_group=azure_conf["RESOURCE_GROUP"],
        workspace_name=azure_conf["WORKSPACE_NAME"],
        datastore_name=azure_conf["DATASTORE_NAME"],
        container_name=azure_conf["CONTAINER_NAME"],
        storage_account_name=azure_conf["STORAGE"]["AccountName"],
        storage_account_key=azure_conf["STORAGE"]["AccountKey"],
        compute_name=azure_conf["COMPUTE_NAME"],
        experiment_name=__file__.split(os.sep)[-1].split(".py")[0],
        # source_directory is whole src directory
        source_directory=os.path.dirname(__file__),
        image_name=azure_conf["IMAGE_NAME"],
        use_gpu=True,
    )
