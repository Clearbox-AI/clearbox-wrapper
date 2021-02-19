import os
import pickle
from typing import Any, Dict, Optional, Union

from loguru import logger
import yaml

from clearbox_wrapper.exceptions import ClearboxWrapperException
from clearbox_wrapper.model import MLMODEL_FILE_NAME, Model
import clearbox_wrapper.pyfunc as pyfunc
from clearbox_wrapper.signature import Signature
from clearbox_wrapper.utils import _get_default_conda_env, _get_flavor_configuration
from clearbox_wrapper.wrapper import add_clearbox_flavor_to_model
from clearbox_wrapper.wrapper import FLAVOR_NAME as cb_flavor_name


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
    mlmodel: Optional[Model] = None,
    serialization_format: str = SERIALIZATION_FORMAT_CLOUDPICKLE,
    signature: Optional[Signature] = None,
    add_clearbox_flavor: bool = False,
    preprocessing_subpath: str = None,
    data_preparation_subpath: str = None,
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
    signature : Optional[Signature], optional
        A model signature describes model input schema. It can be inferred from datasets with
        valid model type (e.g. the training dataset with target column omitted), by default None

    Raises
    ------
    ClearboxWrapperException
        If unrecognized serialization format or model path already exists.
    """
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
    if mlmodel is None:
        mlmodel = Model()

    if signature is not None:
        mlmodel.signature = signature

    model_data_subpath = "model.pkl"

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

    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # `PyFuncModel` only works for sklearn models that define `predict()`.
    if hasattr(sk_model, "predict"):
        pyfunc.add_pyfunc_flavor_to_model(
            mlmodel,
            loader_module="clearbox_wrapper.sklearn",
            model_path=model_data_subpath,
            env=conda_env_subpath,
        )

    if add_clearbox_flavor:
        add_clearbox_flavor_to_model(
            mlmodel,
            loader_module="clearbox_wrapper.sklearn",
            model_path=model_data_subpath,
            env=conda_env_subpath,
            preprocessing=preprocessing_subpath,
            data_preparation=data_preparation_subpath,
        )

    mlmodel.add_flavor(
        FLAVOR_NAME,
        model_path=model_data_subpath,
        sklearn_version=sklearn.__version__,
        serialization_format=serialization_format,
    )

    mlmodel.save(os.path.join(path, MLMODEL_FILE_NAME))


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


def _load_serialized_model(
    serialized_model_path: str, serialization_format: str
) -> Any:
    """Load a serialized (through pickle or cloudpickle) Scikit-Learn model.

    Parameters
    ----------
    serialized_model_path : str
        File path to the Scikit-Learn serialized model.
    serialization_format : str
        Format in which the model was serialized: SERIALIZATION_FORMAT_PICKLE or
        SERIALIZATION_FORMAT_CLOUDPICKLE

    Returns
    -------
    Any
        A Scikit-Learn model.

    Raises
    ------
    ClearboxWrapperException
        If Unrecognized serialization format.
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
    with open(serialized_model_path, "rb") as f:
        # Models serialized with Cloudpickle cannot necessarily be deserialized using Pickle;
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            return pickle.load(f)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle

            return cloudpickle.load(f)


def _load_pyfunc(model_path: str) -> Any:
    """Load Scikit-Learn model as a PyFunc model. This function is called by pyfunc.load_pyfunc.

    Parameters
    ----------
    model_path : str
        File path to the model with sklearn flavor.

    Returns
    -------
    Any
        A Scikit-Learn model.
    """
    try:
        sklearn_flavor_conf = _get_flavor_configuration(
            model_path=model_path, flavor_name=FLAVOR_NAME
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

    pyfunc_flavor_conf = _get_flavor_configuration(
        model_path=model_path, flavor_name=pyfunc.FLAVOR_NAME
    )
    serialized_model_path = os.path.join(model_path, pyfunc_flavor_conf["model_path"])

    return _load_serialized_model(
        serialized_model_path=serialized_model_path,
        serialization_format=serialization_format,
    )


def _load_clearbox(model_path: str) -> Any:
    """Load Scikit-Learn model as a ClearboxWrapper model.

    Parameters
    ----------
    model_path : str
        File path to the model with sklearn flavor.

    Returns
    -------
    Any
        A Scikit-Learn model.
    """
    try:
        sklearn_flavor_conf = _get_flavor_configuration(
            model_path=model_path, flavor_name=FLAVOR_NAME
        )
        serialization_format = sklearn_flavor_conf.get(
            "serialization_format", SERIALIZATION_FORMAT_CLOUDPICKLE
        )
    except ClearboxWrapperException:
        logger.warning(
            "Could not find scikit-learn flavor configuration during model loading process."
            " Assuming 'pickle' serialization format."
        )
        serialization_format = SERIALIZATION_FORMAT_PICKLE

    clearbox_flavor_conf = _get_flavor_configuration(
        model_path=model_path, flavor_name=cb_flavor_name
    )
    serialized_model_path = os.path.join(model_path, clearbox_flavor_conf["model_path"])

    return _load_serialized_model(
        serialized_model_path=serialized_model_path,
        serialization_format=serialization_format,
    )
