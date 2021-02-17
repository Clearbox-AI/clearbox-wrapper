from abc import ABCMeta, abstractmethod
import os
import posixpath
import shutil
import urllib

import cloudpickle
from loguru import logger
import yaml

from clearbox_wrapper.exceptions import ClearboxWrapperException
from clearbox_wrapper.model import MLMODEL_FILE_NAME, Model
from clearbox_wrapper.pyfunc import add_pyfunc_flavor_to_model, FLAVOR_NAME
from clearbox_wrapper.utils import (
    _copy_file_or_tree,
    _get_default_conda_env,
    _get_flavor_configuration,
    TempDir,
)

CONFIG_KEY_ARTIFACTS = "artifacts"
CONFIG_KEY_ARTIFACT_RELATIVE_PATH = "path"
CONFIG_KEY_ARTIFACT_URI = "uri"
CONFIG_KEY_PYTHON_MODEL = "python_model"
CONFIG_KEY_CLOUDPICKLE_VERSION = "cloudpickle_version"

POSTGRES = "postgresql"
MYSQL = "mysql"
SQLITE = "sqlite"
MSSQL = "mssql"

DATABASE_ENGINES = [POSTGRES, MYSQL, SQLITE, MSSQL]

_INVALID_DB_URI_MSG = (
    "Please refer to https://mlflow.org/docs/latest/tracking.html#storage for "
    "format specifications."
)

_UNSUPPORTED_DB_TYPE_MSG = "Supported database engines are {%s}" % ", ".join(
    DATABASE_ENGINES
)

new_model = Model()


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model() <mlflow.pyfunc.save_model>`
             and :func:`log_model() <mlflow.pyfunc.log_model>` when a user-defined subclass of
             :class:`PythonModel` is provided.
    """
    return _get_default_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__)],
        additional_conda_channels=None,
    )


class PythonModel(object, metaclass=ABCMeta):
    """
    Represents a generic Python model that evaluates inputs and produces API-compatible outputs.
    By subclassing :class:`~PythonModel`, users can create customized MLflow models with the
    "python_function" ("pyfunc") flavor, leveraging custom inference logic and artifact
    dependencies.
    """

    def load_context(self, context):
        """
        Loads artifacts from the specified :class:`~PythonModelContext` that can be used by
        :func:`~PythonModel.predict` when evaluating inputs. When loading an MLflow model with
        :func:`~load_pyfunc`, this method is called as soon as the :class:`~PythonModel` is
        constructed.
        The same :class:`~PythonModelContext` will also be available during calls to
        :func:`~PythonModel.predict`, but it may be more efficient to override this method
        and load artifacts from the context at model load time.
        :param context: A :class:`~PythonModelContext` instance containing artifacts that the
                        model can use to perform inference.
        """

    @abstractmethod
    def predict(self, context, model_input):
        """
        Evaluates a pyfunc-compatible input and produces a pyfunc-compatible output.
        For more information about the pyfunc input/output API, see `pyfunc-inference-api`.
        :param context: A :class:`~PythonModelContext` instance containing artifacts that the
                        model can use to perform inference.
        :param model_input: A pyfunc-compatible input for the model to evaluate.
        """


class PythonModelContext(object):
    """
    A collection of artifacts that a :class:`~PythonModel` can use when performing inference.
    :class:`~PythonModelContext` objects are created *implicitly* by the
    :func:`save_model() <mlflow.pyfunc.save_model>` and
    :func:`log_model() <mlflow.pyfunc.log_model>` persistence methods, using the contents
    specified by the ``artifacts`` parameter of these methods.
    """

    def __init__(self, artifacts):
        """
        :param artifacts: A dictionary of ``<name, artifact_path>`` entries, where
        ``artifact_path`` is an absolute filesystem path to a given artifact.
        """
        self._artifacts = artifacts

    @property
    def artifacts(self):
        """
        A dictionary containing ``<name, artifact_path>`` entries, where ``artifact_path`` is an
        absolute filesystem path to the artifact.
        """
        return self._artifacts


def _save_model_with_class_artifacts_params(
    path,
    python_model,
    artifacts=None,
    conda_env=None,
    code_paths=None,
    mlflow_model=new_model,
):
    """
    :param path: The path to which to save the Python model.
    :param python_model: An instance of a subclass of :class:`~PythonModel`. ``python_model``
                        defines how the model loads artifacts and how it performs inference.
    :param artifacts: A dictionary containing ``<name, artifact_uri>`` entries.
                      Remote artifact URIs
                      are resolved to absolute filesystem paths, producing a dictionary of
                      ``<name, absolute_path>`` entries. ``python_model`` can reference these
                      resolved entries as the ``artifacts`` property of the ``context``
                      attribute. If ``None``, no artifacts are added to the model.
    :param conda_env: Either a dictionary representation of a Conda environment or the
                      path to a Conda environment yaml file. If provided, this decsribes the
                      environment this model should be run in. At minimum, it should specify
                      the dependencies
                      contained in :func:`get_default_conda_env()`. If ``None``, the default
                      :func:`get_default_conda_env()` environment is added to the model.
    :param code_paths: A list of local filesystem paths to Python file dependencies (or
                       directories containing file dependencies). These files are *prepended*
                       to the system path before the model is loaded.
    :param mlflow_model: The model configuration to which to add the ``mlflow.pyfunc`` flavor.
    """
    custom_model_config_kwargs = {
        CONFIG_KEY_CLOUDPICKLE_VERSION: cloudpickle.__version__,
    }
    if isinstance(python_model, PythonModel):
        saved_python_model_subpath = "python_model.pkl"
        with open(os.path.join(path, saved_python_model_subpath), "wb") as out:
            cloudpickle.dump(python_model, out)
        custom_model_config_kwargs[CONFIG_KEY_PYTHON_MODEL] = saved_python_model_subpath
    else:
        raise ClearboxWrapperException(
            "`python_model` must be a subclass of `PythonModel`. Instead, found an"
            " object of type: {python_model_type}".format(
                python_model_type=type(python_model)
            )
        )

    if artifacts:
        saved_artifacts_config = {}
        with TempDir() as tmp_artifacts_dir:
            tmp_artifacts_config = {}
            saved_artifacts_dir_subpath = "artifacts"
            for artifact_name, artifact_uri in artifacts.items():
                tmp_artifact_path = _download_artifact_from_uri(
                    artifact_uri=artifact_uri, output_path=tmp_artifacts_dir.path()
                )
                tmp_artifacts_config[artifact_name] = tmp_artifact_path
                saved_artifact_subpath = posixpath.join(
                    saved_artifacts_dir_subpath,
                    os.path.relpath(
                        path=tmp_artifact_path, start=tmp_artifacts_dir.path()
                    ),
                )
                saved_artifacts_config[artifact_name] = {
                    CONFIG_KEY_ARTIFACT_RELATIVE_PATH: saved_artifact_subpath,
                    CONFIG_KEY_ARTIFACT_URI: artifact_uri,
                }

            shutil.move(
                tmp_artifacts_dir.path(),
                os.path.join(path, saved_artifacts_dir_subpath),
            )
        custom_model_config_kwargs[CONFIG_KEY_ARTIFACTS] = saved_artifacts_config

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    saved_code_subpath = None
    if code_paths is not None:
        saved_code_subpath = "code"
        for code_path in code_paths:
            _copy_file_or_tree(src=code_path, dst=path, dst_dir=saved_code_subpath)

    add_pyfunc_flavor_to_model(
        model=mlflow_model,
        loader_module=__name__,
        code=saved_code_subpath,
        env=conda_env_subpath,
        **custom_model_config_kwargs
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def _download_artifact_from_uri(artifact_uri, output_path=None):
    """
    :param artifact_uri: The *absolute* URI of the artifact to download.
    :param output_path: The local filesystem path to which to download the artifact. If
                        unspecified, a local output path will be created.
    """
    parsed_uri = urllib.parse.urlparse(str(artifact_uri))
    prefix = ""
    if parsed_uri.scheme and not parsed_uri.path.startswith("/"):
        # relative path is a special case, urllib does not reconstruct it properly
        prefix = parsed_uri.scheme + ":"
        parsed_uri = parsed_uri._replace(scheme="")

    # For models:/ URIs, it doesn't make sense to initialize a ModelsArtifactRepository with
    # only the model name portion of the URI, then call download_artifacts with the version
    # info.
    if is_models_uri(artifact_uri):
        root_uri = artifact_uri
        artifact_path = ""
    else:
        artifact_path = posixpath.basename(parsed_uri.path)
        parsed_uri = parsed_uri._replace(path=posixpath.dirname(parsed_uri.path))
        root_uri = prefix + urllib.parse.urlunparse(parsed_uri)

    return get_artifact_repository(artifact_uri=root_uri).download_artifacts(
        artifact_path=artifact_path, dst_path=output_path
    )


def is_models_uri(uri):
    return urllib.parse.urlparse(uri).scheme == "models"


def get_artifact_repository(self, artifact_uri):
    """Get an artifact repository from the registry based on the scheme of artifact_uri
    :param store_uri: The store URI. This URI is used to select which artifact repository
                      implementation to instantiate and is passed to the
                      constructor of the implementation.
    :return: An instance of `mlflow.store.ArtifactRepository` that fulfills the artifact URI
             requirements.
    """
    scheme = get_uri_scheme(artifact_uri)
    repository = self._registry.get(scheme)
    if repository is None:
        raise ClearboxWrapperException(
            "Could not find a registered artifact repository for: {}. "
            "Currently registered schemes are: {}".format(
                artifact_uri, list(self._registry.keys())
            )
        )
    return repository(artifact_uri)


def get_uri_scheme(uri_or_path):
    scheme = urllib.parse.urlparse(uri_or_path).scheme
    if any([scheme.lower().startswith(db) for db in DATABASE_ENGINES]):
        return extract_db_type_from_uri(uri_or_path)
    else:
        return scheme


def extract_db_type_from_uri(db_uri):
    """
    Parse the specified DB URI to extract the database type. Confirm the database type is
    supported. If a driver is specified, confirm it passes a plausible regex.
    """
    scheme = urllib.parse.urlparse(db_uri).scheme
    scheme_plus_count = scheme.count("+")

    if scheme_plus_count == 0:
        db_type = scheme
    elif scheme_plus_count == 1:
        db_type, _ = scheme.split("+")
    else:
        error_msg = "Invalid database URI: '%s'. %s" % (db_uri, _INVALID_DB_URI_MSG)
        raise ClearboxWrapperException(error_msg)

    _validate_db_type_string(db_type)

    return db_type


def _validate_db_type_string(db_type):
    """validates db_type parsed from DB URI is supported"""
    if db_type not in DATABASE_ENGINES:
        error_msg = "Invalid database engine: '%s'. '%s'" % (
            db_type,
            _UNSUPPORTED_DB_TYPE_MSG,
        )
        raise ClearboxWrapperException(error_msg)


def _load_pyfunc(model_path):
    pyfunc_config = _get_flavor_configuration(
        model_path=model_path, flavor_name=FLAVOR_NAME
    )

    python_model_cloudpickle_version = pyfunc_config.get(
        CONFIG_KEY_CLOUDPICKLE_VERSION, None
    )
    if python_model_cloudpickle_version is None:
        logger.warning(
            "The version of CloudPickle used to save the model could not be found in"
            " the MLmodel configuration"
        )
    elif python_model_cloudpickle_version != cloudpickle.__version__:
        # CloudPickle does not have a well-defined cross-version compatibility policy. Micro
        # version releases have been known to cause incompatibilities. Therefore, we match
        # on the full library version
        logger.warning(
            "The version of CloudPickle that was used to save the model, `CloudPickle %s`,"
            " differs from the version of CloudPickle that is currently running,"
            " `CloudPickle %s`, and may be incompatible",
            python_model_cloudpickle_version,
            cloudpickle.__version__,
        )

    python_model_subpath = pyfunc_config.get(CONFIG_KEY_PYTHON_MODEL, None)
    if python_model_subpath is None:
        raise ClearboxWrapperException(
            "Python model path was not specified in the model configuration"
        )
    with open(os.path.join(model_path, python_model_subpath), "rb") as f:
        python_model = cloudpickle.load(f)

    artifacts = {}
    for saved_artifact_name, saved_artifact_info in pyfunc_config.get(
        CONFIG_KEY_ARTIFACTS, {}
    ).items():
        artifacts[saved_artifact_name] = os.path.join(
            model_path, saved_artifact_info[CONFIG_KEY_ARTIFACT_RELATIVE_PATH]
        )

    context = PythonModelContext(artifacts=artifacts)
    python_model.load_context(context=context)
    return _PythonModelPyfuncWrapper(python_model=python_model, context=context)


class _PythonModelPyfuncWrapper(object):
    """
    Wrapper class that creates a predict function such that
    predict(model_input: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """

    def __init__(self, python_model, context):
        """
        :param python_model: An instance of a subclass of :class:`~PythonModel`.
        :param context: A :class:`~PythonModelContext` instance containing artifacts that
                        ``python_model`` may use when performing inference.
        """
        self.python_model = python_model
        self.context = context

    def predict(self, model_input):
        return self.python_model.predict(self.context, model_input)
