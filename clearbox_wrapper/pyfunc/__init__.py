from copy import deepcopy

from clearbox_wrapper.model import Model
from clearbox_wrapper.utils import PYTHON_VERSION

FLAVOR_NAME = "python_function"
MAIN = "loader_module"
CODE = "code"
DATA = "data"
ENV = "env"
PY_VERSION = "python_version"


def add_pyfunc_flavor_to_model(
    model: Model,
    loader_module: str,
    data: str = None,
    code=None,
    env: str = None,
    **kwargs
) -> Model:
    """Add Pyfunc flavor to a model configuration. Caller can use this to create a valid
    Pyfunc model flavor out of an existing directory structure. A Pyfunc flavor will be
    added to the flavors list into the MLModel file:
        flavors:
            python_function:
                env: ...
                loader_module: ...
                model_path: ...
                python_version: ...

    Parameters
    ----------
    model : Model
        Existing model.
    loader_module : str
        The module to be used to load the model (e.g. clearbox_wrapper.sklearn)
    data : str, optional
        Path to the model data, by default None.
    code : str, optional
        Path to the code dependencies, by default None.
    env : str, optional
        Path to the Conda environment, by default None.

    Returns
    -------
    Model
        The Model with the new flavor added.
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


__all__ = [
    add_pyfunc_flavor_to_model,
    FLAVOR_NAME,
    MAIN,
    PY_VERSION,
    CODE,
    DATA,
    ENV,
]
