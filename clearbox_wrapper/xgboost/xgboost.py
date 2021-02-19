import os
from typing import Any, Dict, Optional, Union

import yaml

from clearbox_wrapper.exceptions import ClearboxWrapperException
from clearbox_wrapper.model import MLMODEL_FILE_NAME, Model
import clearbox_wrapper.pyfunc as pyfunc
from clearbox_wrapper.signature import Signature
from clearbox_wrapper.utils import _get_default_conda_env, _get_flavor_configuration
from clearbox_wrapper.wrapper import add_clearbox_flavor_to_model
from clearbox_wrapper.wrapper import FLAVOR_NAME as cb_flavor_name


FLAVOR_NAME = "xgboost"


def get_default_xgboost_conda_env() -> Dict:
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
    import xgboost as xgb

    pip_deps = ["xgboost=={}".format(xgb.__version__)]
    return _get_default_conda_env(additional_pip_deps=pip_deps)


def save_xgboost_model(
    xgb_model: Any,
    path: str,
    conda_env: Optional[Union[str, Dict]] = None,
    mlmodel: Optional[Model] = None,
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
    import xgboost as xgb

    if os.path.exists(path):
        raise ClearboxWrapperException("Model path '{}' already exists".format(path))
    os.makedirs(path)

    if mlmodel is None:
        mlmodel = Model()
    if signature is not None:
        mlmodel.signature = signature

    model_data_subpath = "model.xgb"
    xgb_model.save_model(os.path.join(path, model_data_subpath))

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_xgboost_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)

    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    pyfunc.add_pyfunc_flavor_to_model(
        mlmodel,
        loader_module="clearbox_wrapper.xgboost",
        model_path=model_data_subpath,
        env=conda_env_subpath,
    )

    if add_clearbox_flavor:
        add_clearbox_flavor_to_model(
            mlmodel,
            loader_module="clearbox_wrapper.xgboost",
            model_path=model_data_subpath,
            env=conda_env_subpath,
            preprocessing=preprocessing_subpath,
            data_preparation=data_preparation_subpath,
        )

    mlmodel.add_flavor(
        FLAVOR_NAME,
        model_path=model_data_subpath,
        sklearn_version=xgb.__version__,
        data=model_data_subpath,
    )

    mlmodel.save(os.path.join(path, MLMODEL_FILE_NAME))


def _load_model(model_path):
    import xgboost as xgb

    model = xgb.Booster()
    model.load_model(model_path)
    return model


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``xgboost`` flavor.
    """
    return _XGBModelWrapper(_load_model(path))


def _load_clearbox(model_path: str):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``xgboost`` flavor.
    """
    clearbox_flavor_conf = _get_flavor_configuration(
        model_path=model_path, flavor_name=cb_flavor_name
    )
    serialized_model_path = os.path.join(model_path, clearbox_flavor_conf["model_path"])
    return _XGBModelWrapper(_load_model(serialized_model_path))


class _XGBModelWrapper:
    def __init__(self, xgb_model):
        self.xgb_model = xgb_model

    def predict(self, dataframe):
        import xgboost as xgb

        return self.xgb_model.predict(xgb.DMatrix(dataframe))

    def predict_proba(self, dataframe):
        import xgboost as xgb

        return self.xgb_model.predict_proba(xgb.DMatrix(dataframe))
