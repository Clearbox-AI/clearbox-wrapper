from copy import deepcopy

from clearbox_wrapper.utils.environment import PYTHON_VERSION

FLAVOR_NAME = "python_function"
MAIN = "loader_module"
CODE = "code"
DATA = "data"
ENV = "env"
PY_VERSION = "python_version"


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


__all__ = [
    add_to_model,
    FLAVOR_NAME,
    MAIN,
    PY_VERSION,
    CODE,
    DATA,
    ENV,
]
