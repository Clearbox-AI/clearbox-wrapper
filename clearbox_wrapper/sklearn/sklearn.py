import os
import pickle

from loguru import logger
import yaml

from clearbox_wrapper.exceptions import ClearboxWrapperException
from clearbox_wrapper.model.model import MLMODEL_FILE_NAME, Model
from clearbox_wrapper.pyfunc import pyfunc
from clearbox_wrapper.signature.signature import ModelSignature
from clearbox_wrapper.utils.environment import _conda_env
from clearbox_wrapper.utils.model import _get_flavor_configuration


FLAVOR_NAME = "sklearn"

SERIALIZATION_FORMAT_PICKLE = "pickle"
SERIALIZATION_FORMAT_CLOUDPICKLE = "cloudpickle"

SUPPORTED_SERIALIZATION_FORMATS = [
    SERIALIZATION_FORMAT_PICKLE,
    SERIALIZATION_FORMAT_CLOUDPICKLE,
]


def get_default_conda_env(include_cloudpickle=False):
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    import sklearn

    pip_deps = ["scikit-learn=={}".format(sklearn.__version__)]
    if include_cloudpickle:
        import cloudpickle

        pip_deps += ["cloudpickle=={}".format(cloudpickle.__version__)]
    return _conda_env(additional_pip_deps=pip_deps, additional_conda_channels=None)


def save_model(
    sk_model,
    path,
    conda_env=None,
    mlflow_model=None,
    serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE,
    signature: ModelSignature = None,
):
    """
    Save a scikit-learn model to a path on the local file system. Produces an MLflow Model
    containing the following flavors:
        - :py:mod:`mlflow.sklearn`
        - :py:mod:`mlflow.pyfunc`. NOTE: This flavor is only included for scikit-learn models
          that define `predict()`, since `predict()` is required for pyfunc model inference.
    :param sk_model: scikit-learn model to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decsribes the environment
                      this model should be run in. At minimum, it should specify the
                      dependencies contained in :func:`get_default_conda_env()`. If `None`,
                      the default :func:`get_default_conda_env()` environment is added to the
                      model. The following is an *example* dictionary representation of a Conda
                      environment::
                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'scikit-learn=0.19.2'
                            ]
                        }
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    :param serialization_format: The format in which to serialize the model. This should be one
                                of the formats listed in
                                 ``mlflow.sklearn.SUPPORTED_SERIALIZATION_FORMATS``. The
                                 Cloudpickle format,
                                 ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``,
                                 provides better cross-system compatibility by identifying and
                                 packaging code dependencies with the serialized model.
    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be
                      :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with
                      target column omitted) and valid model output (e.g. model predictions
                      generated on the training dataset), for example:
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
    .. code-block:: python
        :caption: Example
        import mlflow.sklearn
        from sklearn.datasets import load_iris
        from sklearn import tree
        iris = load_iris()
        sk_model = tree.DecisionTreeClassifier()
        sk_model = sk_model.fit(iris.data, iris.target)
        # Save the model in cloudpickle format
        # set path to location for persistence
        sk_path_dir_1 = ...
        mlflow.sklearn.save_model(
                sk_model, sk_path_dir_1,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
        # save the model in pickle format
        # set path to location for persistence
        sk_path_dir_2 = ...
        mlflow.sklearn.save_model(sk_model, sk_path_dir_2,
                                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)
    """
    import sklearn

    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise ClearboxWrapperException(
            message=(
                "Unrecognized serialization format: {serialization_format}. Please specify one"
                " of the following supported formats: {supported_formats}.".format(
                    serialization_format=serialization_format,
                    supported_formats=SUPPORTED_SERIALIZATION_FORMATS,
                )
            )
        )

    if os.path.exists(path):
        raise ClearboxWrapperException(
            message="Model path '{}' already exists".format(path)
        )
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature

    model_data_subpath = "model.pkl"
    _save_model(
        sk_model=sk_model,
        output_path=os.path.join(path, model_data_subpath),
        serialization_format=serialization_format,
    )

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env(
            include_cloudpickle=serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE
        )
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # `PyFuncModel` only works for sklearn models that define `predict()`.
    if hasattr(sk_model, "predict"):
        pyfunc.add_to_model(
            mlflow_model,
            loader_module="mlflow.sklearn",
            model_path=model_data_subpath,
            env=conda_env_subpath,
        )
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        pickled_model=model_data_subpath,
        sklearn_version=sklearn.__version__,
        serialization_format=serialization_format,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def _save_model(sk_model, output_path, serialization_format):
    """
    :param sk_model: The scikit-learn model to serialize.
    :param output_path: The file path to which to write the serialized model.
    :param serialization_format: The format in which to serialize the model. This should be
                        one of the following: ``mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE`` or
                        ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``.
    """
    with open(output_path, "wb") as out:
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            pickle.dump(sk_model, out)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle

            cloudpickle.dump(sk_model, out)
        else:
            raise ClearboxWrapperException(
                message="Unrecognized serialization format: {serialization_format}".format(
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
            message=(
                "Unrecognized serialization format: {serialization_format}. Please specify one"
                " of the following supported formats: {supported_formats}.".format(
                    serialization_format=serialization_format,
                    supported_formats=SUPPORTED_SERIALIZATION_FORMATS,
                )
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

    pyfunc_flavor_conf = _get_flavor_configuration(
        model_path=path, flavor_name=pyfunc.FLAVOR_NAME
    )
    path = os.path.join(path, pyfunc_flavor_conf["model_path"])

    return _load_model_from_local_file(
        path=path, serialization_format=serialization_format
    )
