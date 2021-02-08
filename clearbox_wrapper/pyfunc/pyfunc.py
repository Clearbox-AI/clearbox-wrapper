from copy import deepcopy
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import yaml

from clearbox_wrapper.exceptions import ClearboxWrapperException
from clearbox_wrapper.model.model import Model
from clearbox_wrapper.signature.schema import DataType, Schema
from clearbox_wrapper.utils.environment import PYTHON_VERSION


FLAVOR_NAME = "python_function"
MAIN = "loader_module"
CODE = "code"
DATA = "data"
ENV = "env"
PY_VERSION = "python_version"

PyFuncInput = Union[pd.DataFrame, np.ndarray, List[Any], Dict[str, Any]]
PyFuncOutput = Union[pd.DataFrame, pd.Series, np.ndarray, list]


def add_to_model(model, loader_module, data=None, code=None, env=None, **kwargs):
    """
    Add a ``pyfunc`` spec to the model configuration.
    Defines ``pyfunc`` configuration schema. Caller can use this to create a valid ``pyfunc``
    model flavor out of an existing directory structure. For example, other model flavors can
    use this to specify how to use their output as a ``pyfunc``.
    NOTE:
        All paths are relative to the exported model root directory.
    :param model: Existing model.
    :param loader_module: The module to be used to load the model.
    :param data: Path to the model data.
    :param code: Path to the code dependencies.
    :param env: Conda environment.
    :param kwargs: Additional key-value pairs to include in the ``pyfunc`` flavor specification.
                   Values must be YAML-serializable.
    :return: Updated model configuration.
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
    return model.add_flavor(FLAVOR_NAME, **parms)


def _enforce_type(name, values: pd.Series, t: DataType):
    """
    Enforce the input column type matches the declared in model input schema.
    The following type conversions are allowed:
    1. np.object -> string
    2. int -> long (upcast)
    3. float -> double (upcast)
    4. int -> double (safe conversion)
    Any other type mismatch will raise error.
    """
    if values.dtype == np.object and t not in (DataType.binary, DataType.string):
        values = values.infer_objects()

    if t == DataType.string and values.dtype == np.object:
        #  NB: strings are by default parsed and inferred as objects, but it is
        # recommended to use StringDtype extension type if available. See
        #
        # `https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html`
        #
        # for more detail.
        try:
            return values.astype(t.to_pandas(), errors="raise")
        except ValueError:
            raise ClearboxWrapperException(
                "Failed to convert column {0} from type {1} to {2}.".format(
                    name, values.dtype, t
                )
            )

    # NB: Comparison of pandas and numpy data type fails when numpy data type is on the left
    # hand side of the comparison operator. It works, however, if pandas type is on the left
    # hand side. That is because pandas is aware of numpy.
    if t.to_pandas() == values.dtype or t.to_numpy() == values.dtype:
        # The types are already compatible => conversion is not necessary.
        return values

    if t == DataType.binary and values.dtype.kind == t.binary.to_numpy().kind:
        # NB: bytes in numpy have variable itemsize depending on the length of the longest
        # element in the array (column). Since MLflow binary type is length agnostic, we ignore
        # itemsize when matching binary columns.
        return values

    numpy_type = t.to_numpy()
    if values.dtype.kind == numpy_type.kind:
        is_upcast = values.dtype.itemsize <= numpy_type.itemsize
    elif values.dtype.kind == "u" and numpy_type.kind == "i":
        is_upcast = values.dtype.itemsize < numpy_type.itemsize
    elif values.dtype.kind in ("i", "u") and numpy_type == np.float64:
        # allow (u)int => double conversion
        is_upcast = values.dtype.itemsize <= 6
    else:
        is_upcast = False

    if is_upcast:
        return values.astype(numpy_type, errors="raise")
    else:
        # NB: conversion between incompatible types (e.g. floats -> ints or
        # double -> float) are not allowed. While supported by pandas and numpy,
        # these conversions alter the values significantly.
        def all_ints(xs):
            return all([pd.isnull(x) or int(x) == x for x in xs])

        hint = ""
        if (
            values.dtype == np.float64
            and numpy_type.kind in ("i", "u")
            and values.hasnans
            and all_ints(values)
        ):
            hint = (
                " Hint: the type mismatch is likely caused by missing values. "
                "Integer columns in python can not represent missing values and are therefore "
                "encoded as floats. The best way to avoid this problem is to infer the model "
                "schema based on a realistic data sample (training dataset) that includes "
                "missing values. Alternatively, you can declare integer columns as doubles "
                "(float64) whenever these columns may have missing values. See `Handling "
                "Integers With Missing Values <https://www.mlflow.org/docs/latest/models.html#"
                "handling-integers-with-missing-values>`_ for more details."
            )

        raise ClearboxWrapperException(
            "Incompatible input types for column {0}. "
            "Can not safely convert {1} to {2}.{3}".format(
                name, values.dtype, numpy_type, hint
            )
        )


def _enforce_schema(pdf: PyFuncInput, input_schema: Schema):
    """
    Enforce column names and types match the input schema.
    For column names, we check there are no missing columns and reorder the columns to match the
    ordering declared in schema if necessary. Any extra columns are ignored.
    For column types, we make sure the types match schema or can be safely converted to match
    the input schema.
    """
    if isinstance(pdf, (list, np.ndarray, dict)):
        try:
            pdf = pd.DataFrame(pdf)
        except Exception as e:
            message = (
                "This model contains a model signature, which suggests a DataFrame input."
                "There was an error casting the input data to a DataFrame: {0}".format(
                    str(e)
                )
            )
            raise ClearboxWrapperException(message)
    if not isinstance(pdf, pd.DataFrame):
        message = (
            "Expected input to be DataFrame or list. Found: %s" % type(pdf).__name__
        )
        raise ClearboxWrapperException(message)

    if input_schema.has_column_names():
        # make sure there are no missing columns
        col_names = input_schema.column_names()
        expected_names = set(col_names)
        actual_names = set(pdf.columns)
        missing_cols = expected_names - actual_names
        extra_cols = actual_names - expected_names
        # Preserve order from the original columns, since missing/extra columns are likely to
        # be in same order.
        missing_cols = [c for c in col_names if c in missing_cols]
        extra_cols = [c for c in pdf.columns if c in extra_cols]
        if missing_cols:
            message = (
                "Model input is missing columns {0}."
                " Note that there were extra columns: {1}".format(
                    missing_cols, extra_cols
                )
            )
            raise ClearboxWrapperException(message)
    else:
        # The model signature does not specify column names => we can only verify column count.
        if len(pdf.columns) < len(input_schema.columns):
            message = (
                "Model input is missing input columns. The model signature declares "
                "{0} input columns but the provided input only has "
                "{1} columns. Note: the columns were not named in the signature so we can "
                "only verify their count."
            ).format(len(input_schema.columns), len(pdf.columns))
            raise ClearboxWrapperException(message)
        col_names = pdf.columns[: len(input_schema.columns)]
    col_types = input_schema.column_types()
    new_pdf = pd.DataFrame()
    for i, x in enumerate(col_names):
        new_pdf[x] = _enforce_type(x, pdf[x], col_types[i])
    return new_pdf


class PyFuncModel(object):
    """
    MLflow 'python function' model.
    Wrapper around model implementation and metadata. This class is not meant to be constructed
    directly. Instead, instances of this class are constructed and returned from
    py:func:`mlflow.pyfunc.load_model`.
    ``model_impl`` can be any Python object that implements the `Pyfunc interface
    <https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#pyfunc-inference-api>`_, and
    is returned by invoking the model's ``loader_module``.
    ``model_meta`` contains model metadata loaded from the MLmodel file.
    """

    def __init__(self, model_meta: Model, model_impl: Any):
        if not hasattr(model_impl, "predict"):
            raise ClearboxWrapperException(
                "Model implementation is missing required predict method."
            )
        if not model_meta:
            raise ClearboxWrapperException("Model is missing metadata.")
        self._model_meta = model_meta
        self._model_impl = model_impl

    def predict(self, data: PyFuncInput) -> PyFuncOutput:
        """
        Generate model predictions.
        If the model contains signature, enforce the input schema first before calling the model
        implementation with the sanitized input. If the pyfunc model does not include model
        schema, the input is passed to the model implementation as is. See `Model Signature
        Enforcement <https://www.mlflow.org/docs/latest/models.html#signature-enforcement>`_
        for more details."
        :param data: Model input
        :return: Model predictions as one of pandas.DataFrame, pandas.Series, numpy.ndarray
        or list.
        """
        input_schema = self.metadata.get_input_schema()
        if input_schema is not None:
            data = _enforce_schema(data, input_schema)
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
            if (
                hasattr(self._model_meta, "run_id")
                and self._model_meta.run_id is not None
            ):
                info["run_id"] = self._model_meta.run_id
            if (
                hasattr(self._model_meta, "artifact_path")
                and self._model_meta.artifact_path is not None
            ):
                info["artifact_path"] = self._model_meta.artifact_path
            info["flavor"] = self._model_meta.flavors[FLAVOR_NAME]["loader_module"]
        return yaml.safe_dump(
            {"mlflow.pyfunc.loaded_model": info}, default_flow_style=False
        )
