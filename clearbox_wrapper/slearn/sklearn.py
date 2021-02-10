import os
import pickle
from typing import Any, Dict, Optional, Union

from loguru import logger
import yaml

from clearbox_wrapper.exceptions import ClearboxWrapperException
from clearbox_wrapper.model import MLMODEL_FILE_NAME, Model
from clearbox_wrapper.pyfunc import add_pyfunc_flavor_to_model
from clearbox_wrapper.signature.signature import ModelSignature
from clearbox_wrapper.utils.environment import _get_default_conda_env
from clearbox_wrapper.utils.model_utils import _get_flavor_configuration


FLAVOR_NAME = "sklearn"

SERIALIZATION_FORMAT_PICKLE = "pickle"
SERIALIZATION_FORMAT_CLOUDPICKLE = "cloudpickle"

SUPPORTED_SERIALIZATION_FORMATS = [
    SERIALIZATION_FORMAT_PICKLE,
    SERIALIZATION_FORMAT_CLOUDPICKLE,
]


def get_default_sklearn_conda_env(include_cloudpickle: bool = False) -> Dict:
    """Generate the default Conda environment for Scikit-Learn models.

    Parameters
    ----------
    include_cloudpickle : bool, optional
        Whether to include cloudpickle as a environment dependency, by default False.

    Returns
    -------
    Dict
        The default Conda environment for Scikit-Learn models as a dictionary.
    """
    import sklearn

    pip_deps = ["scikit-learn=={}".format(sklearn.__version__)]
    if include_cloudpickle:
        import cloudpickle

        pip_deps += ["cloudpickle=={}".format(cloudpickle.__version__)]
    return _get_default_conda_env(
        additional_pip_deps=pip_deps, additional_conda_channels=None
    )


def save_sklearn_model(
    sk_model: Any,
    path: str,
    conda_env: Optional[Union[str, Dict]] = None,
    serialization_format: str = SERIALIZATION_FORMAT_CLOUDPICKLE,
    signature: Optional[ModelSignature] = None,
):
    """Save a Scikit-Learn model. Produces an MLflow Model containing the following flavors:
        * wrapper.sklearn
        * wrapper.pyfunc. NOTE: This flavor is only included for scikit-learn models
          that define at least `predict()`, since `predict()` is required for pyfunc model
          inference.

    Parameters
    ----------
    sk_model : Any
        A Scikit-Learn model to be saved.
    path : str
        Local path to save the model to.
    conda_env : Optional[Union[str, Dict]], optional
        A dictionary representation of a Conda environment or the path to a Conda environment
        YAML file, by default None. This decsribes the environment this model should be run in.
        If None, the default Conda environment will be added to the model. Example of a
        dictionary representation of a Conda environment:
        {
            'name': 'conda-env',
            'channels': ['defaults'],
            'dependencies': [
                'python=3.7.0',
                'scikit-learn=0.19.2'
            ]
        }
    serialization_format : str, optional
        The format in which to serialize the model. This should be one of the formats listed in
        SUPPORTED_SERIALIZATION_FORMATS. Cloudpickle format, SERIALIZATION_FORMAT_CLOUDPICKLE,
        provides better cross-system compatibility by identifying and packaging code
        dependencies with the serialized model, by default SERIALIZATION_FORMAT_CLOUDPICKLE
    signature : Optional[ModelSignature], optional
        A model signature describes model input schema. It can be inferred from datasets with
        valid model type (e.g. the training dataset with target column omitted), by default None

    Raises
    ------
    ClearboxWrapperException
        If unrecognized serialization format or model path already exists.
    """
    logger.debug(
        "Sono save_model di sklearn, con i seguenti parametri: sk_model={0}, path={1},"
        " conda_env={2}, serialization_format={3}, signature={4}".format(
            sk_model, path, conda_env, serialization_format, signature
        )
    )

    logger.debug(
        "Sono save_model di sklearn, con i seguenti parametri: sk_model={0}, path={1},"
        " conda_env={2}, serialization_format={3}, signature={4}".format(
            type(sk_model),
            type(path),
            type(conda_env),
            type(serialization_format),
            type(signature),
        )
    )
    import sklearn

    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise ClearboxWrapperException(
            "Unrecognized serialization format: {serialization_format}. Please specify one"
            " of the following supported formats: {supported_formats}.".format(
                serialization_format=serialization_format,
                supported_formats=SUPPORTED_SERIALIZATION_FORMATS,
            )
        )
    if os.path.exists(path):
        raise ClearboxWrapperException("Model path '{}' already exists".format(path))

    os.makedirs(path)
    wrapped_model = Model()

    logger.debug(
        "Sono save_model di sklearn, ho creato un nuovo Model: __str__()={0}, to_dict()={1},"
        " to_yaml()={2}, to_json={3}, model={4}".format(
            wrapped_model.__str__(),
            wrapped_model.to_dict(),
            wrapped_model.to_yaml(),
            wrapped_model.to_json(),
            wrapped_model,
        )
    )

    if signature is not None:
        wrapped_model.signature = signature

    model_data_subpath = "model.pkl"

    logger.debug(
        "Sono save_model di sklearn, sto per chiamare _save_model coi seguenti parametri:"
        " sk_model={0}, output_path={1},"
        " serialization_format={2}".format(
            sk_model, os.path.join(path, model_data_subpath), serialization_format
        )
    )
    _serialize_and_save_model(
        sk_model=sk_model,
        output_path=os.path.join(path, model_data_subpath),
        serialization_format=serialization_format,
    )

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_sklearn_conda_env(
            include_cloudpickle=serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE
        )
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    logger.debug(
        "conda_env type: {0}, conda_env: {1}".format(type(conda_env), conda_env)
    )
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # `PyFuncModel` only works for sklearn models that define `predict()`.
    if hasattr(sk_model, "predict"):
        logger.debug(
            "Sono save_model di sklearn, sto per chiamare att_to_model coi seguenti parametri:"
            " wrapped_model={0}, loader_module={1},"
            " model_path={2}, env={3}".format(
                wrapped_model,
                "clearbox_wrapper.slearn.sklearn",
                model_data_subpath,
                conda_env_subpath,
            )
        )
        add_pyfunc_flavor_to_model(
            wrapped_model,
            loader_module="clearbox_wrapper.slearn.sklearn",
            model_path=model_data_subpath,
            env=conda_env_subpath,
        )

    logger.debug("Wrapped model after add_to_model: {}".format(wrapped_model))

    wrapped_model.add_flavor(
        FLAVOR_NAME,
        pickled_model=model_data_subpath,
        sklearn_version=sklearn.__version__,
        serialization_format=serialization_format,
    )

    logger.debug("Wrapped model after second add_to_model: {}".format(wrapped_model))

    wrapped_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def _serialize_and_save_model(
    sk_model: Any, output_path: str, serialization_format: str
) -> None:
    """Serialize and save a Scikit-Learn model to a local file.

    Parameters
    ----------
    sk_model : Any
        The Scikit-Learn model to serialize.
    output_path : str
        The file path to which to write the serialized model (.pkl).
    serialization_format : str
        The format in which to serialize the model. This should be one of the following:
        SERIALIZATION_FORMAT_PICKLE or SERIALIZATION_FORMAT_CLOUDPICKLE.

    Raises
    ------
    ClearboxWrapperException
        Unrecognized serialization format.
    """
    logger.debug("Sono _save_model di sklearn, sto per salvare il modello in model.pkl")

    with open(output_path, "wb") as out:
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            pickle.dump(sk_model, out)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle

            cloudpickle.dump(sk_model, out)
        else:
            raise ClearboxWrapperException(
                "Unrecognized serialization format: {serialization_format}".format(
                    serialization_format=serialization_format
                )
            )


def _load_model_from_local_file(path, serialization_format):
    """Load a scikit-learn model saved as an MLflow artifact on the local file system.
    :param path: Local filesystem path to the MLflow Model saved with the ``sklearn`` flavor
    :param serialization_format: The format in which the model was serialized. This should
                        be one of the following: ``mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE``
                        or ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``.
    """
    # TODO: we could validate the scikit-learn version here
    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise ClearboxWrapperException(
            "Unrecognized serialization format: {serialization_format}. Please specify one"
            " of the following supported formats: {supported_formats}.".format(
                serialization_format=serialization_format,
                supported_formats=SUPPORTED_SERIALIZATION_FORMATS,
            )
        )
    with open(path, "rb") as f:
        # Models serialized with Cloudpickle cannot necessarily be deserialized using Pickle;
        # That's why we check the serialization format of the model before deserializing
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            return pickle.load(f)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle

            return cloudpickle.load(f)


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    :param path: Local filesystem path to the MLflow Model with the ``sklearn`` flavor.
    """
    # In contrast, scikit-learn models saved in versions of MLflow > 1.9.1 do not
    # specify the ``data`` field within the pyfunc flavor configuration. For these newer
    # models, the ``path`` parameter of ``load_pyfunc()`` refers to the top-level MLflow
    # Model directory. In this case, we parse the model path from the MLmodel's pyfunc
    # flavor configuration and attempt to fetch the serialization format from the
    # scikit-learn flavor configuration
    try:
        sklearn_flavor_conf = _get_flavor_configuration(
            model_path=path, flavor_name=FLAVOR_NAME
        )
        serialization_format = sklearn_flavor_conf.get(
            "serialization_format", SERIALIZATION_FORMAT_PICKLE
        )
    except ClearboxWrapperException:
        logger.warning(
            "Could not find scikit-learn flavor configuration during model loading process."
            " Assuming 'pickle' serialization format."
        )
        serialization_format = SERIALIZATION_FORMAT_PICKLE

    logger.debug("path: {}".format(path))
    logger.debug("FLAVOR_NAME: {}".format(FLAVOR_NAME))
    pyfunc_flavor_conf = _get_flavor_configuration(
        model_path=path, flavor_name=FLAVOR_NAME
    )
    logger.debug("pyfunc_flavor_conf: {}".format(pyfunc_flavor_conf))
    path = os.path.join(path, pyfunc_flavor_conf["pickled_model"])

    return _load_model_from_local_file(
        path=path, serialization_format=serialization_format
    )
