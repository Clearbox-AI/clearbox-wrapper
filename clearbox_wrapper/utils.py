import inspect
import os
import shutil
from typing import Any, List
import zipfile


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
