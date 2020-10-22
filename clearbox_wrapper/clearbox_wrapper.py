import os
from typing import Any, Optional, Union, Callable

import cloudpickle

import numpy as np
import pandas as pd

import mlflow.pyfunc


def save_model(
    path: str,
    model: Any,
    preprocessing: Optional[Callable] = None,
    data_cleaning: Optional[Callable] = None,
) -> "ClearboxWrapper":
    wrapped_model = ClearboxWrapper(model, preprocessing, data_cleaning)
    wrapped_model.save(path)

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

    def preprocess(self, data: Any) -> Union[pd.DataFrame, np.ndarray]:
        if self.preprocessing is not None:
            preprocessed_data = (
                self.preprocessing.transform(data)
                if "transform" in dir(self.preprocessing)
                else self.preprocessing(data)
            )
            return preprocessed_data
        else:
            raise Exception("There is no preprocessing for this model.")

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
        return self.model.predict_proba(model_input)

    def save(self, path: str) -> None:
        mlflow.set_tracking_uri(path)
        mlflow.pyfunc.save_model(path=path, python_model=self)
