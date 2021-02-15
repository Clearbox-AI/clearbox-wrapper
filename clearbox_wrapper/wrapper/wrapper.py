import importlib
import os
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger
import numpy as np
import pandas as pd
import yaml

from clearbox_wrapper.exceptions import ClearboxWrapperException
from clearbox_wrapper.model import MLMODEL_FILE_NAME, Model
from clearbox_wrapper.preprocessing import (
    load_serialized_preprocessing,
    Preprocessing,
)
from clearbox_wrapper.signature.signature import infer_signature, Signature
import clearbox_wrapper.slearn.sklearn
from clearbox_wrapper.utils.environment import (
    _get_default_conda_env,
    get_major_minor_py_version,
    PYTHON_VERSION,
)
from clearbox_wrapper.utils.model_utils import get_super_classes_names
from . import DATA, FLAVOR_NAME, MAIN, PREPROCESSING, PY_VERSION
from .model import ClearboxModel

WrapperInput = Union[pd.DataFrame, pd.Series, np.ndarray, List[Any], Dict[str, Any]]
WrapperOutput = Union[pd.DataFrame, pd.Series, np.ndarray, list]


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
        if prepare_data:
            data = self._data_preparation.prepare_data(data)

        if preprocess:
            data = self._preprocessing.preprocess(data)

        if hasattr(self._model_impl, "predict_proba"):
            return self._model_impl.predict_proba(data)
        else:
            return self._model_impl.predict(data)

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
    import dill

    pip_deps = ["dill=={}".format(dill.__version__)]

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
    model_signature: Optional[Signature] = None,
    zip: bool = True,
) -> None:

    logger.debug("Sono save_model di Wrapper.")
    mlmodel = Model()

    if preprocessing is None:
        saved_preprocessing_subpath = None
        if model_signature is not None:
            mlmodel.model_signature = model_signature
        elif input_data is not None:
            logger.debug("input_data is not None")
            model_signature = infer_signature(input_data)
            logger.debug("model_signature:\n{}".format(model_signature))
            mlmodel.model_signature = model_signature
            logger.debug("mlmodel:\n{}".format(mlmodel))
    else:
        logger.debug("preprocessing is not None")
        data_preprocessing = Preprocessing(preprocessing)
        logger.debug("data_preprocessing:\n{}".format(data_preprocessing))
        saved_preprocessing_subpath = "preprocessing.dill"
        if input_data is not None:
            preprocessing_output = data_preprocessing.preprocess(input_data)
            logger.debug("preprocessing_output:\n{}".format(preprocessing_output))
            preprocessing_signature = infer_signature(input_data, preprocessing_output)
            logger.debug("preprocessing_signature:\n{}".format(preprocessing_signature))
            mlmodel.preprocessing_signature = preprocessing_signature
            model_signature = infer_signature(preprocessing_output)
            logger.debug("model_signature:\n{}".format(model_signature))
            mlmodel.model_signature = model_signature
            logger.debug("mlmodel:\n{}".format(mlmodel))

    conda_env = _check_and_get_conda_env(model, additional_deps)
    logger.debug("conda_env:\n{}".format(conda_env))

    logger.debug("Model: {}".format(model))
    model_super_classes = get_super_classes_names(model)
    logger.debug("model_super_classes: {}".format(model_super_classes))
    if any("sklearn" in super_class for super_class in model_super_classes):
        logger.debug("E' un modello Sklearn")
        clearbox_wrapper.slearn.sklearn.save_sklearn_model(
            model,
            path,
            conda_env=conda_env,
            mlmodel=mlmodel,
            add_clearbox_flavor=True,
            preprocessing_subpath=saved_preprocessing_subpath,
        )
        data_preprocessing.save(os.path.join(path, saved_preprocessing_subpath))


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
    logger.debug(
        "Sono load_model, ho ricevuto: model_path={}, suppress_warning={}".format(
            model_path, suppress_warnings
        )
    )

    logger.debug(
        "Sono load_model, sto per chiamare Model.load con questo parametro: {}".format(
            os.path.join(model_path, MLMODEL_FILE_NAME)
        )
    )

    mlmodel = Model.load(os.path.join(model_path, MLMODEL_FILE_NAME))

    logger.debug("Sono load_model, ho caricato model_meta: {}".format(mlmodel))

    clearbox_flavor_configuration = mlmodel.flavors.get(FLAVOR_NAME)

    logger.debug(
        "Sono load_model, ecco pyfunc_flavor_configuration: {}".format(
            clearbox_flavor_configuration
        )
    )

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

    logger.debug(
        "Sono load_model, ecco clearbox_flavor_configuration[MAIN]: {}".format(
            clearbox_flavor_configuration[MAIN]
        )
    )

    data_path = (
        os.path.join(model_path, clearbox_flavor_configuration[DATA])
        if (DATA in clearbox_flavor_configuration)
        else model_path
    )

    model_implementation = importlib.import_module(
        clearbox_flavor_configuration[MAIN]
    )._load_clearbox(data_path)

    logger.debug("Sono load_model, ecco model_impl: {}".format(model_implementation))

    if PREPROCESSING in clearbox_flavor_configuration:
        logger.warning("PORCO DI IDDIO!")
        preprocessing_path = os.path.join(
            model_path, clearbox_flavor_configuration[PREPROCESSING]
        )
        preprocessing = load_serialized_preprocessing(preprocessing_path)
        logger.warning(preprocessing)

    loaded_model = WrapperModel(
        model_meta=mlmodel, model_impl=model_implementation, preprocessing=preprocessing
    )

    logger.info(loaded_model)
    logger.info(type(loaded_model))

    return loaded_model
