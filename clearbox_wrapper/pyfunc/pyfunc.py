import importlib
import os
import sys
from typing import Any, Dict, List, Union

from loguru import logger
import numpy as np
import pandas as pd
import yaml

from clearbox_wrapper.exceptions import ClearboxWrapperException
from clearbox_wrapper.model import MLMODEL_FILE_NAME, Model
from clearbox_wrapper.pyfunc import (
    add_pyfunc_flavor_to_model,
    CODE,
    DATA,
    FLAVOR_NAME,
    MAIN,
    PY_VERSION,
)
from clearbox_wrapper.pyfunc.model import (
    _save_model_with_class_artifacts_params,
    get_default_conda_env,
)
from clearbox_wrapper.schema import DataType, Schema
from clearbox_wrapper.signature import Signature
from clearbox_wrapper.utils import (
    _copy_file_or_tree,
    get_major_minor_py_version,
    PYTHON_VERSION,
)


PyFuncInput = Union[pd.DataFrame, np.ndarray, List[Any], Dict[str, Any]]
PyFuncOutput = Union[pd.DataFrame, pd.Series, np.ndarray, list]
new_model = Model()


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


def load_model(model_path: str, suppress_warnings: bool = False) -> PyFuncModel:
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
    mlmodel = Model.load(os.path.join(model_path, MLMODEL_FILE_NAME))
    pyfunc_flavor_configuration = mlmodel.flavors.get(FLAVOR_NAME)

    if pyfunc_flavor_configuration is None:
        raise ClearboxWrapperException(
            'Model does not have the "{flavor_name}" flavor'.format(
                flavor_name=FLAVOR_NAME
            )
        )

    model_python_version = pyfunc_flavor_configuration.get(PY_VERSION)

    if not suppress_warnings:
        _warn_potentially_incompatible_py_version_if_necessary(
            model_py_version=model_python_version
        )

    if CODE in pyfunc_flavor_configuration and pyfunc_flavor_configuration[CODE]:
        code_path = os.path.join(model_path, pyfunc_flavor_configuration[CODE])
        _add_code_to_system_path(code_path=code_path)

    data_path = (
        os.path.join(model_path, pyfunc_flavor_configuration[DATA])
        if (DATA in pyfunc_flavor_configuration)
        else model_path
    )

    model_implementation = importlib.import_module(
        pyfunc_flavor_configuration[MAIN]
    )._load_pyfunc(data_path)

    return PyFuncModel(model_meta=mlmodel, model_impl=model_implementation)


def _add_code_to_system_path(code_path):
    sys.path = [code_path] + _get_code_dirs(code_path) + sys.path


def _get_code_dirs(src_code_path, dst_code_path=None):
    """
    Obtains the names of the subdirectories contained under the specified source code
    path and joins them with the specified destination code path.
    :param src_code_path: The path of the source code directory for which to list
                          subdirectories.
    :param dst_code_path: The destination directory path to which subdirectory names should be
                          joined.
    """
    if not dst_code_path:
        dst_code_path = src_code_path
    return [
        (os.path.join(dst_code_path, x))
        for x in os.listdir(src_code_path)
        if os.path.isdir(x) and not x == "__pycache__"
    ]


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
    path,
    loader_module=None,
    data_path=None,
    code_path=None,
    conda_env=None,
    mlflow_model=None,
    python_model=None,
    artifacts=None,
    signature: Signature = None,
    **kwargs
):
    """
    save_model(path, loader_module=None, data_path=None, code_path=None, conda_env=None,\
               mlflow_model=Model(), python_model=None, artifacts=None)
    Save a Pyfunc model with custom inference logic and optional data dependencies to a path on
    the local filesystem.
    For information about the workflows that this method supports, please see :ref:`"workflows
    for creating custom pyfunc models" <pyfunc-create-custom-workflows>` and
    :ref:`"which workflow is right for my use case?" <pyfunc-create-custom-selecting-workflow>`.
    Note that the parameters for the second workflow: ``loader_module``, ``data_path`` and the
    parameters for the first workflow: ``python_model``, ``artifacts``, cannot be
    specified together.
    :param path: The path to which to save the Python model.
    :param loader_module: The name of the Python module that is used to load the model
                          from ``data_path``. This module must define a method with the
                          prototype ``_load_pyfunc(data_path)``. If not ``None``,
                          this module and its dependencies must be included in one of
                          the following locations:
                          - The MLflow library.
                          - Package(s) listed in the model's Conda environment, specified by
                            the ``conda_env`` parameter.
                          - One or more of the files specified by the ``code_path`` parameter.
    :param data_path: Path to a file or directory containing model data.
    :param code_path: A list of local filesystem paths to Python file dependencies (or
                      directories containing file dependencies). These files are *prepended* to
                      the system path before the model is loaded.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. This decsribes the environment this model
                      should be run in. If ``python_model`` is not ``None``, the Conda
                      environment must at least specify the dependencies contained in
                      :func:`get_default_conda_env()`. If ``None``, the default
                      :func:`get_default_conda_env()` environment is added to the
                      model. The following is an *example* dictionary representation of a Conda
                      environment::
                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'cloudpickle==0.5.8'
                            ]
                        }
    :param mlflow_model: :py:mod:`mlflow.models.Model` configuration to which to add the
                         **python_function** flavor.
    :param python_model: An instance of a subclass of :class:`~PythonModel`. This class is
                         serialized using the CloudPickle library. Any dependencies of the class
                         should be included in one of the following locations:
                            - The MLflow library.
                            - Package(s) listed in the model's Conda environment, specified by
                              the ``conda_env`` parameter.
                            - One or more of the files specified by the ``code_path`` parameter.
                         Note: If the class is imported from another module, as opposed to being
                         defined in the ``__main__`` scope, the defining module should also be
                         included in one of the listed locations.
    :param artifacts: A dictionary containing ``<name, artifact_uri>`` entries. Remote artifact
                      URIs are resolved to absolute filesystem paths, producing a dictionary of
                      ``<name, absolute_path>`` entries. ``python_model`` can reference these
                      resolved entries as the ``artifacts`` property of the ``context``
                      parameter in :func:`PythonModel.load_context()
                      <mlflow.pyfunc.PythonModel.load_context>`
                      and :func:`PythonModel.predict() <mlflow.pyfunc.PythonModel.predict>`.
                      For example, consider the following ``artifacts`` dictionary::
                        {
                            "my_file": "s3://my-bucket/path/to/my/file"
                        }
                      In this case, the ``"my_file"`` artifact is downloaded from S3. The
                      ``python_model`` can then refer to ``"my_file"`` as an absolute filesystem
                      path via ``context.artifacts["my_file"]``.
                      If ``None``, no artifacts are added to the model.
    :param signature: (Experimental) :py:class:`Signature <mlflow.models.Signature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred
                      <mlflow.models.infer_signature>` from datasets with valid model input
                      (e.g. the training dataset with target column omitted) and valid model
                      output (e.g. model predictions generated on the training dataset),
                      for example:
                      .. code-block:: python
                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of
                          valid model input. The example can be used as a hint of what data to
                          feed the model. The given example will be converted to a Pandas
                          DataFrame and then serialized to json using the Pandas split-oriented
                          format. Bytes are base64-encoded.
    """
    mlflow_model = kwargs.pop("model", mlflow_model)
    if len(kwargs) > 0:
        raise TypeError(
            "save_model() got unexpected keyword arguments: {}".format(kwargs)
        )
    if code_path is not None:
        if not isinstance(code_path, list):
            raise TypeError(
                "Argument code_path should be a list, not {}".format(type(code_path))
            )

    first_argument_set = {
        "loader_module": loader_module,
        "data_path": data_path,
    }
    second_argument_set = {
        "artifacts": artifacts,
        "python_model": python_model,
    }
    first_argument_set_specified = any(
        [item is not None for item in first_argument_set.values()]
    )
    second_argument_set_specified = any(
        [item is not None for item in second_argument_set.values()]
    )
    if first_argument_set_specified and second_argument_set_specified:
        raise ClearboxWrapperException(
            "The following sets of parameters cannot be specified together: {first_set_keys}"
            " and {second_set_keys}. All parameters in one set must be `None`. Instead, found"
            " the following values: {first_set_entries} and {second_set_entries}".format(
                first_set_keys=first_argument_set.keys(),
                second_set_keys=second_argument_set.keys(),
                first_set_entries=first_argument_set,
                second_set_entries=second_argument_set,
            )
        )
    elif (loader_module is None) and (python_model is None):
        msg = (
            "Either `loader_module` or `python_model` must be specified. A `loader_module` "
            "should be a python module. A `python_model` should be a subclass of PythonModel"
        )
        raise ClearboxWrapperException(msg)

    if os.path.exists(path):
        raise ClearboxWrapperException("Path '{}' already exists".format(path))
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature

    if first_argument_set_specified:
        return _save_model_with_loader_module_and_data_path(
            path=path,
            loader_module=loader_module,
            data_path=data_path,
            code_paths=code_path,
            conda_env=conda_env,
            mlflow_model=mlflow_model,
        )
    elif second_argument_set_specified:
        return _save_model_with_class_artifacts_params(
            path=path,
            python_model=python_model,
            artifacts=artifacts,
            conda_env=conda_env,
            code_paths=code_path,
            mlflow_model=mlflow_model,
        )


def _save_model_with_loader_module_and_data_path(
    path,
    loader_module,
    data_path=None,
    code_paths=None,
    conda_env=None,
    mlflow_model=new_model,
):
    """
    Export model as a generic Python function model.
    :param path: The path to which to save the Python model.
    :param loader_module: The name of the Python module that is used to load the model
                          from ``data_path``. This module must define a method with the
                          prototype ``_load_pyfunc(data_path)``.
    :param data_path: Path to a file or directory containing model data.
    :param code_paths: A list of local filesystem paths to Python file dependencies (or
                      directories containing file dependencies). These files are *prepended*
                      to the system path before the model is loaded.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decsribes the environment
                      this model should be run in.
    :return: Model configuration containing model info.
    """

    code = None
    data = None

    if data_path is not None:
        model_file = _copy_file_or_tree(src=data_path, dst=path, dst_dir="data")
        data = model_file

    if code_paths is not None:
        for code_path in code_paths:
            _copy_file_or_tree(src=code_path, dst=path, dst_dir="code")
        code = "code"

    conda_env_subpath = "mlflow_env.yml"
    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    add_pyfunc_flavor_to_model(
        mlflow_model,
        loader_module=loader_module,
        code=code,
        data=data,
        env=conda_env_subpath,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))
    return mlflow_model
