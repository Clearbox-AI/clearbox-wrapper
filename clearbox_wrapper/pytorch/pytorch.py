from distutils.version import LooseVersion
import importlib
import os
import posixpath
import shutil
from typing import Any, Dict, Optional, Union

import cloudpickle
from loguru import logger
import numpy as np
import pandas as pd
import yaml

from clearbox_wrapper.exceptions import ClearboxWrapperException
from clearbox_wrapper.model import MLMODEL_FILE_NAME, Model
import clearbox_wrapper.pyfunc as pyfunc
from clearbox_wrapper.pytorch import pickle_module as clearbox_pytorch_pickle_module
from clearbox_wrapper.signature import Signature
from clearbox_wrapper.utils import (
    _copy_file_or_tree,
    _get_default_conda_env,
    _get_flavor_configuration,
    TempDir,
)
from clearbox_wrapper.wrapper import add_clearbox_flavor_to_model


FLAVOR_NAME = "pytorch"

_SERIALIZED_TORCH_MODEL_FILE_NAME = "model.pth"
_TORCH_STATE_DICT_FILE_NAME = "state_dict.pth"
_PICKLE_MODULE_INFO_FILE_NAME = "pickle_module_info.txt"
_EXTRA_FILES_KEY = "extra_files"
_REQUIREMENTS_FILE_KEY = "requirements_file"


def get_default_pytorch_conda_env() -> Dict:
    import torch
    import torchvision

    pip_deps = [
        "cloudpickle=={}".format(cloudpickle.__version__),
        "pytorch=={}".format(torch.__version__),
        "torchvision=={}".format(torchvision.__version__),
    ]

    return _get_default_conda_env(additional_pip_deps=pip_deps)


def save_pytorch_model(
    pytorch_model: Any,
    path: str,
    conda_env: Optional[Union[str, Dict]] = None,
    mlmodel: Optional[Model] = None,
    signature: Optional[Signature] = None,
    add_clearbox_flavor: bool = False,
    preprocessing_subpath: str = None,
    data_preparation_subpath: str = None,
    code_paths=None,
    pickle_module=None,
    requirements_file=None,
    extra_files=None,
    **kwargs
):
    import torch

    pickle_module = pickle_module or clearbox_pytorch_pickle_module
    if not isinstance(pytorch_model, torch.nn.Module):
        raise TypeError("Argument 'pytorch_model' should be a torch.nn.Module")
    if code_paths is not None:
        if not isinstance(code_paths, list):
            raise TypeError(
                "Argument code_paths should be a list, not {}".format(type(code_paths))
            )

    if os.path.exists(path):
        raise ClearboxWrapperException("Model path '{}' already exists".format(path))
    os.makedirs(path)

    if mlmodel is None:
        mlmodel = Model()
    if signature is not None:
        mlmodel.signature = signature

    model_data_subpath = "data"
    model_data_path = os.path.join(path, model_data_subpath)
    os.makedirs(model_data_path)

    # Persist the pickle module name as a file in the model's `data` directory. This is
    # necessary because the `data` directory is the only available parameter to
    # `_load_pyfunc`, and it does not contain the MLmodel configuration; therefore,
    # it is not sufficient to place the module name in the MLmodel
    #
    # TODO: Stop persisting this information to the filesystem once we have a mechanism for
    # supplying the MLmodel configuration to `mlflow.pytorch._load_pyfunc`
    pickle_module_path = os.path.join(model_data_path, _PICKLE_MODULE_INFO_FILE_NAME)
    with open(pickle_module_path, "w") as f:
        f.write(pickle_module.__name__)

    # Save pytorch model
    model_path = os.path.join(model_data_path, _SERIALIZED_TORCH_MODEL_FILE_NAME)
    if isinstance(pytorch_model, torch.jit.ScriptModule):
        torch.jit.ScriptModule.save(pytorch_model, model_path)
    else:
        torch.save(pytorch_model, model_path, pickle_module=pickle_module, **kwargs)

    torchserve_artifacts_config = {}

    if requirements_file:
        if not isinstance(requirements_file, str):
            raise TypeError("Path to requirements file should be a string")

        with TempDir() as tmp_requirements_dir:

            rel_path = os.path.basename(requirements_file)
            torchserve_artifacts_config[_REQUIREMENTS_FILE_KEY] = {"path": rel_path}
            shutil.move(tmp_requirements_dir.path(rel_path), path)

    if extra_files:
        torchserve_artifacts_config[_EXTRA_FILES_KEY] = []
        if not isinstance(extra_files, list):
            raise TypeError("Extra files argument should be a list")

        with TempDir() as tmp_extra_files_dir:
            for extra_file in extra_files:
                rel_path = posixpath.join(
                    _EXTRA_FILES_KEY,
                    os.path.basename(extra_file),
                )
                torchserve_artifacts_config[_EXTRA_FILES_KEY].append({"path": rel_path})
            shutil.move(
                tmp_extra_files_dir.path(),
                posixpath.join(path, _EXTRA_FILES_KEY),
            )

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_pytorch_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    if code_paths is not None:
        code_dir_subpath = "code"
        for code_path in code_paths:
            _copy_file_or_tree(src=code_path, dst=path, dst_dir=code_dir_subpath)
    else:
        code_dir_subpath = None

    mlmodel.add_flavor(
        FLAVOR_NAME,
        model_data=model_data_subpath,
        pytorch_version=torch.__version__,
        **torchserve_artifacts_config,
    )

    pyfunc.add_pyfunc_flavor_to_model(
        mlmodel,
        loader_module="clearbox_wrapper.pytorch",
        data=model_data_subpath,
        pickle_module_name=pickle_module.__name__,
        code=code_dir_subpath,
        env=conda_env_subpath,
    )

    if add_clearbox_flavor:
        add_clearbox_flavor_to_model(
            mlmodel,
            loader_module="clearbox_wrapper.pytorch",
            data=model_data_subpath,
            pickle_module_name=pickle_module.__name__,
            code=code_dir_subpath,
            env=conda_env_subpath,
            preprocessing=preprocessing_subpath,
            data_preparation=data_preparation_subpath,
        )

    mlmodel.save(os.path.join(path, MLMODEL_FILE_NAME))


def _load_model(path, **kwargs):
    """
    :param path: The path to a serialized PyTorch model.
    :param kwargs: Additional kwargs to pass to the PyTorch ``torch.load`` function.
    """
    import torch

    if os.path.isdir(path):
        # `path` is a directory containing a serialized PyTorch model and a text file containing
        # information about the pickle module that should be used by PyTorch to load it
        model_path = os.path.join(path, "model.pth")
        pickle_module_path = os.path.join(path, _PICKLE_MODULE_INFO_FILE_NAME)
        with open(pickle_module_path, "r") as f:
            pickle_module_name = f.read()
        if (
            "pickle_module" in kwargs
            and kwargs["pickle_module"].__name__ != pickle_module_name
        ):
            logger.warning(
                "Attempting to load the PyTorch model with a pickle module, '%s', that does not"
                " match the pickle module that was used to save the model: '%s'.",
                kwargs["pickle_module"].__name__,
                pickle_module_name,
            )
        else:
            try:
                kwargs["pickle_module"] = importlib.import_module(pickle_module_name)
            except ImportError as exc:
                raise ClearboxWrapperException(
                    message=(
                        "Failed to import the pickle module that was used to save the PyTorch"
                        " model. Pickle module name: `{pickle_module_name}`".format(
                            pickle_module_name=pickle_module_name
                        )
                    )
                ) from exc

    else:
        model_path = path

    if LooseVersion(torch.__version__) >= LooseVersion("1.5.0"):
        return torch.load(model_path, **kwargs)
    else:
        try:
            # load the model as an eager model.
            return torch.load(model_path, **kwargs)
        except Exception:
            # If fails, assume the model as a scripted model
            return torch.jit.load(model_path)


def load_model(model_path, **kwargs):
    """
    Load a PyTorch model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model, for example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.

    :param kwargs: kwargs to pass to ``torch.load`` method.
    :return: A PyTorch model.

    .. code-block:: python
        :caption: Example

        import torch
        import mlflow.pytorch

        # Class defined here
        class LinearNNModel(torch.nn.Module):
            ...

        # Initialize our model, criterion and optimizer
        ...

        # Training loop
        ...
        # Log the model
        with mlflow.start_run() as run:
            mlflow.pytorch.log_model(model, "model")

        # Inference after loading the logged model
        model_uri = "runs:/{}/model".format(run.info.run_id)
        loaded_model = mlflow.pytorch.load_model(model_uri)
        for x in [4.0, 6.0, 30.0]:
            X = torch.Tensor([[x]])
            y_pred = loaded_model(X)
            print("predict X: {}, y_pred: {:.2f}".format(x, y_pred.data.item()))

    .. code-block:: text
        :caption: Output

        predict X: 4.0, y_pred: 7.57
        predict X: 6.0, y_pred: 11.64
        predict X: 30.0, y_pred: 60.48
    """
    import torch

    try:
        pyfunc_conf = _get_flavor_configuration(
            model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME
        )
    except ClearboxWrapperException:
        pyfunc_conf = {}
    code_subpath = pyfunc_conf.get(pyfunc.CODE)
    if code_subpath is not None:
        pyfunc._add_code_to_system_path(
            code_path=os.path.join(model_path, code_subpath)
        )

    pytorch_conf = _get_flavor_configuration(
        model_path=model_path, flavor_name=FLAVOR_NAME
    )
    if torch.__version__ != pytorch_conf["pytorch_version"]:
        logger.warning(
            "Stored model version '%s' does not match installed PyTorch version '%s'",
            pytorch_conf["pytorch_version"],
            torch.__version__,
        )
    torch_model_artifacts_path = os.path.join(model_path, pytorch_conf["model_data"])
    return _load_model(path=torch_model_artifacts_path, **kwargs)


def _load_pyfunc(path, **kwargs):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``pytorch`` flavor.
    """
    return _PyTorchWrapper(_load_model(path, **kwargs))


def _load_clearbox(path, **kwargs):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``pytorch`` flavor.
    """
    return _PyTorchWrapper(_load_model(path, **kwargs))


class _PyTorchWrapper(object):
    """
    Wrapper class that creates a predict function such that
    predict(data: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """

    def __init__(self, pytorch_model):
        self.pytorch_model = pytorch_model

    def predict(self, data, device="cpu"):
        import torch

        if isinstance(data, pd.DataFrame):
            inp_data = data.values.astype(np.float32)
        elif isinstance(data, np.ndarray):
            inp_data = data
        elif isinstance(data, (list, dict)):
            raise TypeError(
                "The PyTorch flavor does not support List or Dict input types. "
                "Please use a pandas.DataFrame or a numpy.ndarray"
            )
        else:
            raise TypeError("Input data should be pandas.DataFrame or numpy.ndarray")

        self.pytorch_model.to(device)
        self.pytorch_model.eval()
        with torch.no_grad():
            input_tensor = torch.from_numpy(inp_data).to(device)
            preds = self.pytorch_model(input_tensor.float())
            if not isinstance(preds, torch.Tensor):
                raise TypeError(
                    "Expected PyTorch model to output a single output tensor, "
                    "but got output of type '{}'".format(type(preds))
                )
            if isinstance(data, pd.DataFrame):
                predicted = pd.DataFrame(preds.numpy())
                predicted.index = data.index
            else:
                predicted = preds.numpy()
            return predicted
