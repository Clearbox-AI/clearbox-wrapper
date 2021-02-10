import inspect
import os
import shutil
from typing import Any, Dict, List
import zipfile

from loguru import logger

from clearbox_wrapper.exceptions import ClearboxWrapperException
from clearbox_wrapper.model import MLMODEL_FILE_NAME, Model


def _get_flavor_configuration(model_path: str, flavor_name: str) -> Dict:
    """Get the configuration for a specified flavor of a model.

    Parameters
    ----------
    model_path : str
        Path to the model directory.
    flavor_name : str
        Name of the flavor configuration to load.

    Returns
    -------
    Dict
        Flavor configuration as a dictionary.

    Raises
    ------
    ClearboxWrapperException
        If it couldn't find a MLmodel file or if the model doesn't contain
        the specified flavor.
    """
    mlmodel_path = os.path.join(model_path, MLMODEL_FILE_NAME)
    logger.debug(
        "Sono _get_flavor_configuration, mlmodel_path: {}".format(mlmodel_path)
    )
    if not os.path.exists(mlmodel_path):
        raise ClearboxWrapperException(
            'Could not find an "{}" configuration file at "{}"'.format(
                MLMODEL_FILE_NAME, model_path
            )
        )

    mlmodel = Model.load(mlmodel_path)

    logger.debug("Sono _get_flavor_configuration, mlmodel: {}".format(mlmodel))

    if flavor_name not in mlmodel.flavors:
        raise ClearboxWrapperException(
            'Model does not have the "{}" flavor'.format(flavor_name)
        )
    flavor_configuration_dict = mlmodel.flavors[flavor_name]
    logger.debug(
        "Sono _get_flavor_configuration, conf: {}".format(flavor_configuration_dict)
    )
    return flavor_configuration_dict


def get_super_classes_names(instance_or_class: Any) -> List[str]:
    """Given an instance or a class, computes and returns a list of its superclasses.

    Parameters
    ----------
    instance_or_class : Any
        An instance of an object or a class.

    Returns
    -------
    List[str]
        List of superclasses names strings.
    """
    super_class_names_list = []
    if not inspect.isclass(instance_or_class):
        instance_or_class = instance_or_class.__class__
    super_classes_tuple = inspect.getmro(instance_or_class)
    for super_class in super_classes_tuple:
        super_class_name = (
            str(super_class).replace("'", "").replace("<class ", "").replace(">", "")
        )
        super_class_names_list.append(super_class_name)
    return super_class_names_list


def zip_directory(directory_path: str) -> None:
    """Given a directory path, zip the directory.

    Parameters
    ----------
    directory_path : str
        Directory path
    """
    zip_object = zipfile.ZipFile(directory_path + ".zip", "w", zipfile.ZIP_DEFLATED)
    root_len = len(directory_path) + 1
    for base, _dirs, files in os.walk(directory_path):
        for file in files:
            fn = os.path.join(base, file)
            zip_object.write(fn, fn[root_len:])
    shutil.rmtree(directory_path)
