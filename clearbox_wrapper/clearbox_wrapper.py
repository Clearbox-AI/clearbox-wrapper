import os
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cloudpickle
import mlflow.keras
import mlflow.pyfunc
import mlflow.pytorch
from mlflow.utils.environment import _mlflow_conda_env
import numpy as np
import pandas as pd


def _check_and_get_conda_env(model: Any, additional_deps: List = None) -> Dict:
    pip_deps = ["cloudpickle=={}".format(cloudpickle.__version__)]

    if additional_deps is not None:
        pip_deps += additional_deps

    if "__getstate__" in dir(model) and "_sklearn_version" in model.__getstate__():
        pip_deps.append(
            "scikit-learn=={}".format(model.__getstate__()["_sklearn_version"])
        )
    if "xgb" in model.__class__.__name__.lower():
        import xgboost

        pip_deps.append("xgboost=={}".format(xgboost.__version__))
    if "keras" in str(model.__class__):
        import tensorflow

        pip_deps.append("tensorflow=={}".format(tensorflow.__version__))

    if "torch" in str(model.__class__):
        import torch

        pip_deps.append("torch=={}".format(torch.__version__))

    unique_pip_deps = [dep.split("==")[0] for dep in pip_deps]
    if len(unique_pip_deps) > len(set(unique_pip_deps)):
        raise ValueError(
            "Multiple occurences of the same dependency is not allowed: {}".format(
                pip_deps
            )
        )

    return _mlflow_conda_env(additional_pip_deps=pip_deps)


def save_model(
    path: str,
    model: Any,
    preprocessing: Optional[Callable] = None,
    data_cleaning: Optional[Callable] = None,
    additional_deps: List = None,
) -> "ClearboxWrapper":

    conda_env = _check_and_get_conda_env(model, additional_deps)

    if "keras" in str(model.__class__):
        with TemporaryDirectory() as tmp_dir:
            keras_model_path = os.path.join(tmp_dir, "keras_model")
            mlflow.keras.save_model(model, keras_model_path)
            artifacts = {"keras_model": keras_model_path}
            wrapped_model = ClearboxWrapper(None, preprocessing, data_cleaning)
            wrapped_model.save(path, conda_env=conda_env, artifacts=artifacts)
    elif "torch" in str(model.__class__):
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

        def preprocessing_function(
            data: Any, _preprocessing=preprocessing
        ) -> Union[pd.DataFrame, np.ndarray]:
            preprocessed_data = (
                _preprocessing.transform(data)
                if "transform" in dir(_preprocessing)
                else _preprocessing(data)
            )
            return preprocessed_data

        saved_preprocessing_subpath = "preprocessing.pkl"
        with open(
            os.path.join(path, saved_preprocessing_subpath), "wb"
        ) as preprocessing_output_file:
            cloudpickle.dump(preprocessing_function, preprocessing_output_file)

    if data_cleaning is not None:

        def data_cleaning_function(
            data: Any, _data_cleaning=data_cleaning
        ) -> Union[pd.DataFrame, np.ndarray]:
            cleaned_data = (
                _data_cleaning.transform(data)
                if "transform" in dir(_data_cleaning)
                else _data_cleaning(data)
            )
            return cleaned_data

        saved_data_cleaning_subpath = "data_cleaning.pkl"
        with open(
            os.path.join(path, saved_data_cleaning_subpath), "wb"
        ) as data_cleaning_output_file:
            cloudpickle.dump(data_cleaning_function, data_cleaning_output_file)

    return wrapped_model


def load_model(path: str) -> Any:
    loaded_model = mlflow.pyfunc.load_model(path)
    return loaded_model


def load_model_preprocessing(path: str) -> Tuple:
    loaded_model = mlflow.pyfunc.load_model(path)

    saved_preprocessing_path = os.path.join(path, "preprocessing.pkl")
    if os.path.isfile(saved_preprocessing_path):
        with open(saved_preprocessing_path, "rb") as preprocessing_file:
            loaded_preprocessing = cloudpickle.load(preprocessing_file)
    else:
        raise FileNotFoundError("There is no 'preprocessing' for this model.")

    return loaded_model, loaded_preprocessing


def load_model_preprocessing_data_cleaning(path: str) -> Tuple:
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
    ) -> None:
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
            self.model = mlflow.pytorch.load_model(context.artifacts["pytorch_model"])

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

        if context is not None and "pytorch_model" in context.artifacts:
            self.model.eval()
            return self.model(model_input)
        elif "predict_proba" in dir(self.model):
            return self.model.predict_proba(model_input)
        else:
            return self.model.predict(model_input)

    def save(self, path: str, conda_env: Dict = None, artifacts: Dict = None) -> None:
        mlflow.set_tracking_uri(path)
        mlflow.pyfunc.save_model(
            path=path, python_model=self, artifacts=artifacts, conda_env=conda_env
        )
