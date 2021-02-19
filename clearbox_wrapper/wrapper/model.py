from abc import ABCMeta, abstractmethod

import dill

from clearbox_wrapper.utils import _get_default_conda_env

dill.settings["recurse"] = True


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model() <mlflow.pyfunc.save_model>`
             and :func:`log_model() <mlflow.pyfunc.log_model>` when a user-defined subclass of
             :class:`PythonModel` is provided.
    """
    return _get_default_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["dill=={}".format(dill.__version__)],
        additional_conda_channels=None,
    )


class ClearboxModel(object, metaclass=ABCMeta):
    @abstractmethod
    def prepare_data(self, data):
        pass

    @abstractmethod
    def preprocess_data(self, data):
        pass

    @abstractmethod
    def predict(self, model_input):
        pass

    @abstractmethod
    def predict_proba(self, model_input):
        pass


class _ModelWrapper(object):
    """
    Wrapper class that creates a predict function such that
    predict(model_input: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """

    def __init__(self, wrapper_model):
        """
        :param python_model: An instance of a subclass of :class:`~PythonModel`.
        :param context: A :class:`~PythonModelContext` instance containing artifacts that
                        ``python_model`` may use when performing inference.
        """
        self.wrapper_model = wrapper_model

    def prepare_data(self, data):
        return self.wrapper_model.prepare_data(data)

    def preprocess_data(self, data):
        return self.wrapper_model.preprocess_data(data)

    def predict(self, model_input):
        return self.wrapper_model.predict(model_input)
