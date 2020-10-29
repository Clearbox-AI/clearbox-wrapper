import os
from typing import Any, List, Dict, Optional, Union, Callable

from tempfile import TemporaryDirectory

import cloudpickle

import numpy as np
import pandas as pd

import mlflow.pyfunc
import mlflow.keras
import mlflow.pytorch
from mlflow.utils.environment import _mlflow_conda_env


def _check_and_get_conda_env(
    model: Any,
    additional_conda_deps: List = None,
    additional_pip_deps: List = None,
    additional_conda_channels: List = None,
) -> Dict:
    pip_deps = ["cloudpickle=={}".format(cloudpickle.__version__)]
    conda_deps = []

    if additional_pip_deps is not None:
        pip_deps += additional_pip_deps
    if additional_conda_deps is not None:
        conda_deps += additional_conda_deps

    if additional_conda_channels is not None and len(additional_conda_channels) > len(
        set(additional_conda_channels)
    ):
        raise ValueError("Each element of 'additional_conda_channels' must be unique.")

    if "__getstate__" in dir(model) and "_sklearn_version" in model.__getstate__():
        conda_deps.append(
            "scikit-learn={}".format(model.__getstate__()["_sklearn_version"])
        )
    if "xgb" in model.__class__.__name__.lower():
        import xgboost

        pip_deps.append("xgboost=={}".format(xgboost.__version__))
    if "keras" in str(model.__class__):
        import tensorflow

        pip_deps.append("tensorflow=={}".format(tensorflow.__version__))

    unique_conda_deps = [dep.split("=")[0] for dep in conda_deps]
    if len(unique_conda_deps) > len(set(unique_conda_deps)):
        raise ValueError(
            "Multiple occurences of a conda dependency is not allowed: {}".format(
                conda_deps
            )
        )

    unique_pip_deps = [dep.split("==")[0] for dep in pip_deps]
    if len(unique_pip_deps) > len(set(unique_pip_deps)):
        raise ValueError(
            "Multiple occurences of a pip dependency is not allowed: {}".format(
                pip_deps
            )
        )

    conda_pip_common_deps = set(unique_conda_deps).intersection(set(unique_pip_deps))
    if len(conda_pip_common_deps) > 0:
        raise ValueError(
            "Some deps have been passed for both conda and pip: {}".format(
                conda_pip_common_deps
            )
        )

    return _mlflow_conda_env(
        additional_conda_deps=conda_deps,
        additional_pip_deps=pip_deps,
        additional_conda_channels=additional_conda_channels,
    )


def save_model(
    path: str,
    model: Any,
    preprocessing: Optional[Callable] = None,
    data_cleaning: Optional[Callable] = None,
    additional_conda_deps: List = None,
    additional_pip_deps: List = None,
    additional_conda_channels: List = None,
) -> "ClearboxWrapper":

    conda_env = _check_and_get_conda_env(
        model, additional_conda_deps, additional_pip_deps, additional_conda_channels
    )

    if "keras" in str(model.__class__):
        with TemporaryDirectory() as tmp_dir:
            keras_model_path = os.path.join(tmp_dir, "keras_model")
            mlflow.keras.save_model(model, keras_model_path)
            artifacts = {"keras_model": keras_model_path}
            wrapped_model = ClearboxWrapper(None, preprocessing, data_cleaning)
            wrapped_model.save(path, conda_env=conda_env, artifacts=artifacts)
    elif "torch" in str(model.__class__):
        print("PORCODDIO!")
        with TemporaryDirectory() as tmp_dir:
            pytorch_model_path = os.path.join(tmp_dir, "pytorch_model")
            mlflow.pytorch.save_model(model, pytorch_model_path)
            artifacts = {"pytorch_model": pytorch_model_path}
            wrapped_model = ClearboxWrapper(None, preprocessing, data_cleaning)
            wrapped_model.save(path, conda_env=conda_env, artifacts=artifacts)
    else:
        wrapped_model = ClearboxWrapper(model, preprocessing, data_cleaning)
        wrapped_model.save(path, conda_env=conda_env)

    if preprocessing is not None:

        def preprocessing_function(data: Any) -> Union[pd.DataFrame, np.ndarray]:
            preprocessed_data = (
                preprocessing.transform(data)
                if "transform" in dir(preprocessing)
                else preprocessing(data)
            )
            return preprocessed_data

        saved_preprocessing_subpath = "preprocessing.pkl"
        with open(
            os.path.join(path, saved_preprocessing_subpath), "wb"
        ) as preprocessing_output_file:
            cloudpickle.dump(preprocessing_function, preprocessing_output_file)

    if data_cleaning is not None:

        def data_cleaning_function(data: Any) -> Union[pd.DataFrame, np.ndarray]:
            cleaned_data = (
                data_cleaning.transform(data)
                if "transform" in dir(data_cleaning)
                else data_cleaning(data)
            )
            return cleaned_data

        saved_data_cleaning_subpath = "data_cleaning.pkl"
        with open(
            os.path.join(path, saved_data_cleaning_subpath), "wb"
        ) as data_cleaning_output_file:
            cloudpickle.dump(data_cleaning_function, data_cleaning_output_file)

    return wrapped_model


def load_model(path: str):
    loaded_model = mlflow.pyfunc.load_model(path)
    return loaded_model


def load_model_preprocessing(path: str):
    loaded_model = mlflow.pyfunc.load_model(path)

    saved_preprocessing_path = os.path.join(path, "preprocessing.pkl")
    if os.path.isfile(saved_preprocessing_path):
        with open(saved_preprocessing_path, "rb") as preprocessing_file:
            loaded_preprocessing = cloudpickle.load(preprocessing_file)
    else:
        raise FileNotFoundError("There is no 'preprocessing' for this model.")

    return loaded_model, loaded_preprocessing


def load_model_preprocessing_data_cleaning(path: str):
    loaded_model = mlflow.pyfunc.load_model(path)

    saved_preprocessing_path = os.path.join(path, "preprocessing.pkl")
    if os.path.isfile(saved_preprocessing_path):
        with open(saved_preprocessing_path, "rb") as preprocessing_file:
            loaded_preprocessing = cloudpickle.load(preprocessing_file)
    else:
        raise FileNotFoundError("There is no 'preprocessing' for this model.")

    saved_data_cleaning_path = os.path.join(path, "data_cleaning.pkl")
    if os.path.isfile(saved_data_cleaning_path):
        with open(saved_data_cleaning_path, "rb") as data_cleaning_file:
            loaded_data_cleaning = cloudpickle.load(data_cleaning_file)
    else:
        raise FileNotFoundError("There is no 'data_cleaning' for this model.")

    return loaded_model, loaded_preprocessing, loaded_data_cleaning


class ClearboxWrapper(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        model: Any,
        preprocessing: Optional[Callable] = None,
        data_cleaning: Optional[Callable] = None,
    ) -> "ClearboxWrapper":
        if preprocessing is None and data_cleaning is not None:
            raise ValueError(
                "Attribute 'preprocessing' is None but attribute "
                "'data_cleaning' is not None. If you have a single step "
                "preprocessing, pass it as attribute 'preprocessing'"
            )
        self.model = model
        self.preprocessing = preprocessing
        self.data_cleaning = data_cleaning

    def load_context(self, context):
        if "keras_model" in context.artifacts:
            self.model = mlflow.keras.load_model(context.artifacts["keras_model"])
        elif "pytorch_model" in context.artifacts:
            print("PORCALAMADONNA!")
            self.model = mlflow.pytorch.load_model(context.artifacts["pytorch_model"])
            print(dir(self.model))

    def predict(self, context=None, model_input=None):
        if self.data_cleaning is not None:
            model_input = (
                self.data_cleaning.transform(model_input)
                if "transform" in dir(self.data_cleaning)
                else self.data_cleaning(model_input)
            )
        if self.preprocessing is not None:
            model_input = (
                self.preprocessing.transform(model_input)
                if "transform" in dir(self.preprocessing)
                else self.preprocessing(model_input)
            )
        if "predict_proba" in dir(self.model):
            return self.model.predict_proba(model_input)
        else:
            return self.model.predict(model_input)

    def save(self, path: str, conda_env: Dict = None, artifacts: Dict = None) -> None:
        mlflow.set_tracking_uri(path)
        print(artifacts)
        mlflow.pyfunc.save_model(
            path=path, python_model=self, artifacts=artifacts, conda_env=conda_env
        )
