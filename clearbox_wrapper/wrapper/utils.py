from copy import deepcopy
import os
import shutil
import zipfile

from clearbox_wrapper.model import Model
from clearbox_wrapper.utils import PYTHON_VERSION

FLAVOR_NAME = "clearbox"
MAIN = "loader_module"
CODE = "code"
DATA = "data"
ENV = "env"
PREPROCESSING = "preprocessing_path"
DATA_PREPARATION = "data_preparation_path"
PY_VERSION = "python_version"


def add_clearbox_flavor_to_model(
    model: Model,
    loader_module: str,
    data: str = None,
    code=None,
    env: str = None,
    preprocessing: str = None,
    data_preparation: str = None,
    **kwargs,
) -> Model:
    """Add Clearbox flavor to a model configuration. Caller can use this to create a valid
    Clearbox model flavor out of an existing directory structure. A Clearbox flavor will be
    added to the flavors list into the MLModel file:
        flavors:
            clearbox:
                env: ...
                loader_module: ...
                model_path: ...
                python_version: ...

    Parameters
    ----------
    model : Model
        Existing model.
    loader_module : str
        The module to be used to load the model (e.g. clearbox_wrapper.sklearn)
    data : str, optional
        Path to the model data, by default None.
    code : str, optional
        Path to the code dependencies, by default None.
    env : str, optional
        Path to the Conda environment, by default None.

    Returns
    -------
    Model
        The Model with the new flavor added.
    """
    parms = deepcopy(kwargs)

    parms[MAIN] = loader_module
    parms[PY_VERSION] = PYTHON_VERSION
    if code:
        parms[CODE] = code
    if data:
        parms[DATA] = data
    if env:
        parms[ENV] = env
    if preprocessing:
        parms[PREPROCESSING] = preprocessing
    if data_preparation:
        parms[DATA_PREPARATION] = data_preparation
    return model.add_flavor(FLAVOR_NAME, **parms)


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
