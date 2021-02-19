import importlib
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger
import numpy as np
import pandas as pd
import yaml

from clearbox_wrapper.data_preparation import (
    DataPreparation,
    load_serialized_data_preparation,
)
from clearbox_wrapper.exceptions import ClearboxWrapperException
from clearbox_wrapper.keras import save_keras_model
from clearbox_wrapper.model import MLMODEL_FILE_NAME, Model
from clearbox_wrapper.preprocessing import (
    load_serialized_preprocessing,
    Preprocessing,
)
from clearbox_wrapper.pytorch import save_pytorch_model
from clearbox_wrapper.signature import infer_signature
from clearbox_wrapper.sklearn import save_sklearn_model
from clearbox_wrapper.utils import (
    _get_default_conda_env,
    get_major_minor_py_version,
    get_super_classes_names,
    PYTHON_VERSION,
)
from clearbox_wrapper.xgboost import save_xgboost_model
from .model import ClearboxModel
from .utils import (
    DATA,
    DATA_PREPARATION,
    FLAVOR_NAME,
    MAIN,
    PREPROCESSING,
    PY_VERSION,
    zip_directory,
)

WrapperInput = Union[pd.DataFrame, pd.Series, np.ndarray, List[Any], Dict[str, Any]]
WrapperOutput = Union[pd.DataFrame, pd.Series, np.ndarray, list]
logger.add(sys.stdout, backtrace=False, diagnose=False)


class WrapperModel(ClearboxModel):
    def __init__(
        self,
        model_meta: Model,
        model_impl: Any,
        preprocessing: Any = None,
        data_preparation: Any = None,
    ):
        if not hasattr(model_impl, "predict"):
            raise ClearboxWrapperException(
                "Model implementation is missing required predict method."
            )
        if not model_meta:
            raise ClearboxWrapperException("Model is missing metadata.")
        if data_preparation is not None and preprocessing is None:
            raise ValueError(
                "Attribute 'preprocessing' is None but attribute "
                "'data_preparation' is not None. If you have a single step "
                "preprocessing, pass it as attribute 'preprocessing'"
            )

        self._model_meta = model_meta
        self._model_impl = model_impl
        self._preprocessing = preprocessing
        self._data_preparation = data_preparation

    def prepare_data(self, data: WrapperInput) -> WrapperOutput:
        if self._data_preparation is None:
            raise ClearboxWrapperException("This model has no data preparation.")
        return self._data_preparation.prepare_data(data)

    def preprocess_data(self, data: WrapperInput) -> WrapperOutput:
        if self._preprocessing is None:
            raise ClearboxWrapperException("This model has no preprocessing.")
        return self._preprocessing.preprocess(data)

    def predict(
        self, data: WrapperInput, preprocess: bool = True, prepare_data: bool = True
    ) -> WrapperOutput:
        if prepare_data and self._data_preparation is not None:
            data = self._data_preparation.prepare_data(data)
        elif not prepare_data:
            logger.warning(
                "This model has data preparation and you're bypassing it,"
                " this can lead to unexpected results."
            )

        if preprocess and self._preprocessing is not None:
            data = self._preprocessing.preprocess(data)
        elif not preprocess:
            logger.warning(
                "This model has preprocessing and you're bypassing it,"
                " this can lead to unexpected results."
            )

        return self._model_impl.predict(data)

    def predict_proba(
        self, data: WrapperInput, preprocess: bool = True, prepare_data: bool = True
    ) -> WrapperOutput:
        if not hasattr(self._model_impl, "predict_proba"):
            raise ClearboxWrapperException("This model has no predict_proba method.")

        if prepare_data and self._data_preparation is not None:
            data = self._data_preparation.prepare_data(data)
        elif not prepare_data:
            logger.warning(
                "This model has data preparation and you're bypassing it,"
                " this can lead to unexpected results."
            )

        if preprocess and self._preprocessing is not None:
            data = self._preprocessing.preprocess(data)
        elif not preprocess:
            logger.warning(
                "This model has preprocessing and you're bypassing it,"
                " this can lead to unexpected results."
            )

        return self._model_impl.predict_proba(data)

    @property
    def metadata(self):
        """Model metadata."""
        if self._model_meta is None:
            raise ClearboxWrapperException("Model is missing metadata.")
        return self._model_meta

    def __repr__(self):
        info = {}
        if self._model_meta is not None:
            info["flavor"] = self._model_meta.flavors[FLAVOR_NAME]["loader_module"]
        return yaml.safe_dump({"wrapper.loaded_model": info}, default_flow_style=False)


def _check_and_get_conda_env(model: Any, additional_deps: List = None) -> Dict:
    import cloudpickle

    pip_deps = ["cloudpickle=={}".format(cloudpickle.__version__)]

    if additional_deps is not None:
        pip_deps += additional_deps

    model_super_classes = get_super_classes_names(model)

    # order of ifs matters
    if any("xgboost" in super_class for super_class in model_super_classes):
        import xgboost

        pip_deps.append("xgboost=={}".format(xgboost.__version__))
    elif any("sklearn" in super_class for super_class in model_super_classes):
        pip_deps.append(
            "scikit-learn=={}".format(model.__getstate__()["_sklearn_version"])
        )
    elif any("keras" in super_class for super_class in model_super_classes):
        import tensorflow

        pip_deps.append("tensorflow=={}".format(tensorflow.__version__))
    elif any("torch" in super_class for super_class in model_super_classes):
        import torch

        pip_deps.append("torch=={}".format(torch.__version__))

    unique_pip_deps = [dep.split("==")[0] for dep in pip_deps]
    if len(unique_pip_deps) > len(set(unique_pip_deps)):
        raise ValueError(
            "Multiple occurences of the same dependency is not allowed: {}".format(
                pip_deps
            )
        )

    return _get_default_conda_env(additional_pip_deps=pip_deps)


def _warn_potentially_incompatible_py_version_if_necessary(model_py_version=None):
    """
    Compares the version of Python that was used to save a given model with the version
    of Python that is currently running. If a major or minor version difference is detected,
    logs an appropriate warning.
    """
    if model_py_version is None:
        logger.warning(
            "The specified model does not have a specified Python version. It may be"
            " incompatible with the version of Python that is currently running: Python %s",
            PYTHON_VERSION,
        )
    elif get_major_minor_py_version(model_py_version) != get_major_minor_py_version(
        PYTHON_VERSION
    ):
        logger.warning(
            "The version of Python that the model was saved in, `Python %s`, differs"
            " from the version of Python that is currently running, `Python %s`,"
            " and may be incompatible",
            model_py_version,
            PYTHON_VERSION,
        )


def save_model(
    path: str,
    model: Any,
    input_data: Optional[WrapperInput] = None,
    preprocessing: Optional[Callable] = None,
    data_preparation: Optional[Callable] = None,
    additional_deps: Optional[List] = None,
    zip: bool = True,
) -> None:

    mlmodel = Model()
    saved_preprocessing_subpath = None
    saved_data_preparation_subpath = None

    if data_preparation is not None and preprocessing is None:
        raise ValueError(
            "Attribute 'preprocessing' is None but attribute "
            "'data_preparation' is not None. If you have a single step "
            "preprocessing, pass it as attribute 'preprocessing'"
        )

    if data_preparation and preprocessing:
        preparation = DataPreparation(data_preparation)
        data_preprocessing = Preprocessing(preprocessing)
        saved_data_preparation_subpath = "data_preparation.pkl"
        saved_preprocessing_subpath = "preprocessing.pkl"
        if input_data is not None:
            data_preparation_output = preparation.prepare_data(input_data)
            preprocessing_output = data_preprocessing.preprocess(
                data_preparation_output
            )
            data_preparation_signature = infer_signature(
                input_data, data_preparation_output
            )
            preprocessing_signature = infer_signature(
                data_preparation_output, preprocessing_output
            )
            model_signature = infer_signature(preprocessing_output)
            mlmodel.preparation_signature = data_preparation_signature
            mlmodel.preprocessing_signature = preprocessing_signature
            mlmodel.model_signature = model_signature
    elif preprocessing:
        data_preprocessing = Preprocessing(preprocessing)
        saved_preprocessing_subpath = "preprocessing.pkl"
        if input_data is not None:
            preprocessing_output = data_preprocessing.preprocess(input_data)
            preprocessing_signature = infer_signature(input_data, preprocessing_output)
            model_signature = infer_signature(preprocessing_output)
            mlmodel.preprocessing_signature = preprocessing_signature
            mlmodel.model_signature = model_signature
    elif input_data is not None:
        model_signature = infer_signature(input_data)
        mlmodel.model_signature = model_signature

    conda_env = _check_and_get_conda_env(model, additional_deps)
    model_super_classes = get_super_classes_names(model)

    if any("sklearn" in super_class for super_class in model_super_classes):
        save_sklearn_model(
            model,
            path,
            conda_env=conda_env,
            mlmodel=mlmodel,
            add_clearbox_flavor=True,
            preprocessing_subpath=saved_preprocessing_subpath,
            data_preparation_subpath=saved_data_preparation_subpath,
        )
    elif any("xgboost" in super_class for super_class in model_super_classes):
        save_xgboost_model(
            model,
            path,
            conda_env=conda_env,
            mlmodel=mlmodel,
            add_clearbox_flavor=True,
            preprocessing_subpath=saved_preprocessing_subpath,
            data_preparation_subpath=saved_data_preparation_subpath,
        )
    elif any("keras" in super_class for super_class in model_super_classes):
        save_keras_model(
            model,
            path,
            conda_env=conda_env,
            mlmodel=mlmodel,
            add_clearbox_flavor=True,
            preprocessing_subpath=saved_preprocessing_subpath,
            data_preparation_subpath=saved_data_preparation_subpath,
        )
    elif any("torch" in super_class for super_class in model_super_classes):
        save_pytorch_model(
            model,
            path,
            conda_env=conda_env,
            mlmodel=mlmodel,
            add_clearbox_flavor=True,
            preprocessing_subpath=saved_preprocessing_subpath,
            data_preparation_subpath=saved_data_preparation_subpath,
        )

    if preprocessing:
        data_preprocessing.save(os.path.join(path, saved_preprocessing_subpath))
    if data_preparation:
        preparation.save(os.path.join(path, saved_data_preparation_subpath))
    if zip:
        zip_directory(path)


def load_model(model_path: str, suppress_warnings: bool = False) -> WrapperModel:
    """Load a model that has python_function flavor.

    Parameters
    ----------
    model_path : str
        Filepath of the model directory.
    suppress_warnings : bool, optional
        If Fatal, non-fatal warning messages associated with the model loading process
        will be emitted, by default True

    Returns
    -------
    PyFuncModel
        A python_function model.

    Raises
    ------
    ClearboxWrapperException
        If the model does not have the python_function flavor.
    """
    preprocessing = None
    data_preparation = None

    mlmodel = Model.load(os.path.join(model_path, MLMODEL_FILE_NAME))
    clearbox_flavor_configuration = mlmodel.flavors.get(FLAVOR_NAME)

    if clearbox_flavor_configuration is None:
        raise ClearboxWrapperException(
            'Model does not have the "{flavor_name}" flavor'.format(
                flavor_name=FLAVOR_NAME
            )
        )

    model_python_version = clearbox_flavor_configuration.get(PY_VERSION)

    if not suppress_warnings:
        _warn_potentially_incompatible_py_version_if_necessary(
            model_py_version=model_python_version
        )

    data_path = (
        os.path.join(model_path, clearbox_flavor_configuration[DATA])
        if (DATA in clearbox_flavor_configuration)
        else model_path
    )

    model_implementation = importlib.import_module(
        clearbox_flavor_configuration[MAIN]
    )._load_clearbox(data_path)

    if PREPROCESSING in clearbox_flavor_configuration:
        preprocessing_path = os.path.join(
            model_path, clearbox_flavor_configuration[PREPROCESSING]
        )
        preprocessing = load_serialized_preprocessing(preprocessing_path)

    if DATA_PREPARATION in clearbox_flavor_configuration:
        data_preparation_path = os.path.join(
            model_path, clearbox_flavor_configuration[DATA_PREPARATION]
        )
        data_preparation = load_serialized_data_preparation(data_preparation_path)

    loaded_model = WrapperModel(
        model_meta=mlmodel,
        model_impl=model_implementation,
        preprocessing=preprocessing,
        data_preparation=data_preparation,
    )

    return loaded_model
