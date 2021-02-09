import inspect
import os
import shutil
from typing import Any, List
import zipfile

from clearbox_wrapper.exceptions import ClearboxWrapperException
from clearbox_wrapper.model import MLMODEL_FILE_NAME, Model


def _get_flavor_configuration(model_path, flavor_name):
    """
    Obtains the configuration for the specified flavor from the specified
    MLflow model path. If the model does not contain the specified flavor,
    an exception will be thrown.
    :param model_path: The path to the root directory of the MLflow model for which to load
                       the specified flavor configuration.
    :param flavor_name: The name of the flavor configuration to load.
    :return: The flavor configuration as a dictionary.
    """
    model_configuration_path = os.path.join(model_path, MLMODEL_FILE_NAME)
    if not os.path.exists(model_configuration_path):
        raise ClearboxWrapperException(
            'Could not find an "{model_file}" configuration file at "{model_path}"'.format(
                model_file=MLMODEL_FILE_NAME, model_path=model_path
            )
        )

    model_conf = Model.load(model_configuration_path)
    if flavor_name not in model_conf.flavors:
        raise ClearboxWrapperException(
            'Model does not have the "{flavor_name}" flavor'.format(
                flavor_name=flavor_name
            )
        )
    conf = model_conf.flavors[flavor_name]
    return conf


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
