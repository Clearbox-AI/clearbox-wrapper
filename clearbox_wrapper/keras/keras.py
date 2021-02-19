from distutils.version import LooseVersion
import importlib
import os
from typing import Any, Dict, Optional, Union

import pandas as pd
import yaml

from clearbox_wrapper.exceptions import ClearboxWrapperException
from clearbox_wrapper.model import MLMODEL_FILE_NAME, Model
import clearbox_wrapper.pyfunc as pyfunc
from clearbox_wrapper.signature import Signature
from clearbox_wrapper.utils import _get_default_conda_env, _get_flavor_configuration
from clearbox_wrapper.wrapper import add_clearbox_flavor_to_model


FLAVOR_NAME = "keras"

_CUSTOM_OBJECTS_SAVE_PATH = "custom_objects.pickle"
_KERAS_MODULE_SPEC_PATH = "keras_module.txt"
_KERAS_SAVE_FORMAT_PATH = "save_format.txt"
_MODEL_SAVE_PATH = "model"


def get_default_keras_conda_env(include_cloudpickle=False, keras_module=None) -> Dict:
    import tensorflow as tf

    pip_deps = []
    if keras_module is None:
        import keras

        keras_module = keras

    if keras_module.__name__ == "keras":
        pip_deps.append("keras=={}".format(keras_module.__version__))

    if include_cloudpickle:
        import cloudpickle

        pip_deps.append("cloudpickle=={}".format(cloudpickle.__version__))

    pip_deps.append("tensorflow=={}".format(tf.__version__))

    # Tensorflow<2.4 does not work with h5py>=3.0.0
    # see https://github.com/tensorflow/tensorflow/issues/44467
    if LooseVersion(tf.__version__) < LooseVersion("2.4"):
        pip_deps.append("h5py<3.0.0")

    return _get_default_conda_env(additional_pip_deps=pip_deps)


def _save_custom_objects(path, custom_objects):
    """
    Save custom objects dictionary to a cloudpickle file so a model can be easily loaded later.

    :param path: An absolute path that points to the data directory within /path/to/model.
    :param custom_objects: Keras ``custom_objects`` is a dictionary mapping
                           names (strings) to custom classes or functions to be considered
                           during deserialization. MLflow saves these custom layers using
                           CloudPickle and restores them automatically when the model is
                           loaded with :py:func:`mlflow.keras.load_model` and
                           :py:func:`mlflow.pyfunc.load_model`.
    """
    import cloudpickle

    custom_objects_path = os.path.join(path, _CUSTOM_OBJECTS_SAVE_PATH)
    with open(custom_objects_path, "wb") as out_f:
        cloudpickle.dump(custom_objects, out_f)


def save_keras_model(
    keras_model: Any,
    path: str,
    conda_env: Optional[Union[str, Dict]] = None,
    mlmodel: Optional[Model] = None,
    signature: Optional[Signature] = None,
    add_clearbox_flavor: bool = False,
    preprocessing_subpath: str = None,
    data_preparation_subpath: str = None,
    keras_module: str = None,
    custom_objects=None,
    **kwargs
):
    if keras_module is None:

        def _is_plain_keras(model):
            try:
                import keras

                if LooseVersion(keras.__version__) < LooseVersion("2.2.0"):
                    import keras.engine

                    return isinstance(model, keras.engine.Model)
                else:
                    # NB: Network is the first parent with save method
                    import keras.engine.network

                    return isinstance(model, keras.engine.network.Network)
            except ImportError:
                return False

        def _is_tf_keras(model):
            try:
                # NB: Network is not exposed in tf.keras, we check for Model instead.
                import tensorflow.keras.models

                return isinstance(model, tensorflow.keras.models.Model)
            except ImportError:
                return False

        if _is_plain_keras(keras_model):
            keras_module = importlib.import_module("keras")
        elif _is_tf_keras(keras_model):
            keras_module = importlib.import_module("tensorflow.keras")
        else:
            raise ClearboxWrapperException(
                "Unable to infer keras module from the model, please specify "
                "which keras module ('keras' or 'tensorflow.keras') is to be "
                "used to save and load the model."
            )
    elif type(keras_module) == str:
        keras_module = importlib.import_module(keras_module)

    if os.path.exists(path):
        raise ClearboxWrapperException("Model path '{}' already exists".format(path))

    data_subpath = "data"
    data_path = os.path.join(path, data_subpath)
    os.makedirs(data_path)

    if mlmodel is None:
        mlmodel = Model()
    if signature is not None:
        mlmodel.signature = signature

    if custom_objects is not None:
        _save_custom_objects(data_path, custom_objects)

    # save keras module spec to path/data/keras_module.txt
    with open(os.path.join(data_path, _KERAS_MODULE_SPEC_PATH), "w") as f:
        f.write(keras_module.__name__)

    # Use the SavedModel format if `save_format` is unspecified
    save_format = kwargs.get("save_format", "tf")

    # save keras save_format to path/data/save_format.txt
    with open(os.path.join(data_path, _KERAS_SAVE_FORMAT_PATH), "w") as f:
        f.write(save_format)

    # save keras model
    # To maintain prior behavior, when the format is HDF5, we save
    # with the h5 file extension. Otherwise, model_path is a directory
    # where the saved_model.pb will be stored (for SavedModel format)
    file_extension = ".h5" if save_format == "h5" else ""
    model_subpath = os.path.join(data_subpath, _MODEL_SAVE_PATH)
    model_path = os.path.join(path, model_subpath) + file_extension
    keras_model.save(model_path, **kwargs)

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_keras_conda_env(
            include_cloudpickle=custom_objects is not None, keras_module=keras_module
        )
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)

    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    mlmodel.add_flavor(
        FLAVOR_NAME,
        keras_module=keras_module.__name__,
        keras_version=keras_module.__version__,
        save_format=save_format,
        data=data_subpath,
    )

    pyfunc.add_pyfunc_flavor_to_model(
        mlmodel,
        loader_module="clearbox_wrapper.keras",
        data=data_subpath,
        env=conda_env_subpath,
    )

    if add_clearbox_flavor:
        add_clearbox_flavor_to_model(
            mlmodel,
            loader_module="clearbox_wrapper.keras",
            data=data_subpath,
            env=conda_env_subpath,
            preprocessing=preprocessing_subpath,
            data_preparation=data_preparation_subpath,
        )

    mlmodel.save(os.path.join(path, MLMODEL_FILE_NAME))


def _load_model(model_path, keras_module, save_format, **kwargs):
    keras_models = importlib.import_module(keras_module.__name__ + ".models")
    custom_objects = kwargs.pop("custom_objects", {})
    custom_objects_path = None
    if os.path.isdir(model_path):
        if os.path.isfile(os.path.join(model_path, _CUSTOM_OBJECTS_SAVE_PATH)):
            custom_objects_path = os.path.join(model_path, _CUSTOM_OBJECTS_SAVE_PATH)
        model_path = os.path.join(model_path, _MODEL_SAVE_PATH)
    if custom_objects_path is not None:
        import cloudpickle

        with open(custom_objects_path, "rb") as in_f:
            pickled_custom_objects = cloudpickle.load(in_f)
            pickled_custom_objects.update(custom_objects)
            custom_objects = pickled_custom_objects

    # If the save_format is HDF5, then we save with h5 file
    # extension to align with prior behavior of mlflow logging
    if save_format == "h5":
        model_path = model_path + ".h5"

    from distutils.version import StrictVersion

    if save_format == "h5" and StrictVersion(
        keras_module.__version__.split("-")[0]
    ) >= StrictVersion("2.2.3"):
        # NOTE: Keras 2.2.3 does not work with unicode paths in python2. Pass in h5py.
        # File instead of string to avoid issues.
        import h5py

        with h5py.File(os.path.abspath(model_path), "r") as model_path:
            return keras_models.load_model(
                model_path, custom_objects=custom_objects, **kwargs
            )
    else:
        # NOTE: Older versions of Keras only handle filepath.
        return keras_models.load_model(
            model_path, custom_objects=custom_objects, **kwargs
        )


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``keras`` flavor.
    """
    import tensorflow as tf

    if os.path.isfile(os.path.join(path, _KERAS_MODULE_SPEC_PATH)):
        with open(os.path.join(path, _KERAS_MODULE_SPEC_PATH), "r") as f:
            keras_module = importlib.import_module(f.read())
    else:
        import keras

        keras_module = keras

    # By default, we assume the save_format is h5 for backwards compatibility
    save_format = "h5"
    save_format_path = os.path.join(path, _KERAS_SAVE_FORMAT_PATH)
    if os.path.isfile(save_format_path):
        with open(save_format_path, "r") as f:
            save_format = f.read()

    # In SavedModel format, if we don't compile the model
    should_compile = save_format == "tf"
    K = importlib.import_module(keras_module.__name__ + ".backend")
    if keras_module.__name__ == "tensorflow.keras" or K.backend() == "tensorflow":
        if LooseVersion(tf.__version__) < LooseVersion("2.0.0"):
            graph = tf.Graph()
            sess = tf.Session(graph=graph)
            # By default tf backed models depend on the global graph and session.
            # We create an use new Graph and Session and store them with the model
            # This way the model is independent on the global state.
            with graph.as_default():
                with sess.as_default():  # pylint:disable=not-context-manager
                    K.set_learning_phase(0)
                    m = _load_model(
                        path,
                        keras_module=keras_module,
                        save_format=save_format,
                        compile=should_compile,
                    )
                    return _KerasModelWrapper(m, graph, sess)
        else:
            K.set_learning_phase(0)
            m = _load_model(
                path,
                keras_module=keras_module,
                save_format=save_format,
                compile=should_compile,
            )
            return _KerasModelWrapper(m, None, None)

    else:
        raise ClearboxWrapperException("Unsupported backend '%s'" % K._BACKEND)


def _load_clearbox(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``keras`` flavor.
    """
    import tensorflow as tf

    if os.path.isfile(os.path.join(path, _KERAS_MODULE_SPEC_PATH)):
        with open(os.path.join(path, _KERAS_MODULE_SPEC_PATH), "r") as f:
            keras_module = importlib.import_module(f.read())
    else:
        import keras

        keras_module = keras

    # By default, we assume the save_format is h5 for backwards compatibility
    save_format = "h5"
    save_format_path = os.path.join(path, _KERAS_SAVE_FORMAT_PATH)
    if os.path.isfile(save_format_path):
        with open(save_format_path, "r") as f:
            save_format = f.read()

    # In SavedModel format, if we don't compile the model
    should_compile = save_format == "tf"
    K = importlib.import_module(keras_module.__name__ + ".backend")
    if keras_module.__name__ == "tensorflow.keras" or K.backend() == "tensorflow":
        if LooseVersion(tf.__version__) < LooseVersion("2.0.0"):
            graph = tf.Graph()
            sess = tf.Session(graph=graph)
            # By default tf backed models depend on the global graph and session.
            # We create an use new Graph and Session and store them with the model
            # This way the model is independent on the global state.
            with graph.as_default():
                with sess.as_default():  # pylint:disable=not-context-manager
                    K.set_learning_phase(0)
                    m = _load_model(
                        path,
                        keras_module=keras_module,
                        save_format=save_format,
                        compile=should_compile,
                    )
                    return _KerasModelWrapper(m, graph, sess)
        else:
            K.set_learning_phase(0)
            m = _load_model(
                path,
                keras_module=keras_module,
                save_format=save_format,
                compile=should_compile,
            )
            return _KerasModelWrapper(m, None, None)

    else:
        raise ClearboxWrapperException("Unsupported backend '%s'" % K._BACKEND)


def load_model(model_path, **kwargs):
    flavor_conf = _get_flavor_configuration(
        model_path=model_path, flavor_name=FLAVOR_NAME
    )
    keras_module = importlib.import_module(flavor_conf.get("keras_module", "keras"))
    keras_model_artifacts_path = os.path.join(
        model_path, flavor_conf.get("data", _MODEL_SAVE_PATH)
    )
    # For backwards compatibility, we assume h5 when the save_format is absent
    save_format = flavor_conf.get("save_format", "h5")
    return _load_model(
        model_path=keras_model_artifacts_path,
        keras_module=keras_module,
        save_format=save_format,
        **kwargs
    )


class _KerasModelWrapper:
    def __init__(self, keras_model, graph, sess):
        self.keras_model = keras_model
        self._graph = graph
        self._sess = sess

    def predict(self, data):
        def _predict(data):
            if isinstance(data, pd.DataFrame):
                predicted = pd.DataFrame(self.keras_model.predict(data.values))
                predicted.index = data.index
            else:
                predicted = self.keras_model.predict(data)
            return predicted

        # In TensorFlow < 2.0, we use a graph and session to predict
        if self._graph is not None:
            with self._graph.as_default():
                with self._sess.as_default():
                    predicted = _predict(data)
        # In TensorFlow >= 2.0, we do not use a graph and session to predict
        else:
            predicted = _predict(data)
        return predicted
