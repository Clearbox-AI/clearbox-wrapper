import os
from typing import Any

from clearbox_wrapper.exceptions import ClearboxWrapperException
from clearbox_wrapper.generic_model import GenericModel


def save(
    path: str,
    model: Any,
    preprocessing: Any = None,
    data_preparation: Any = None,
    artifacts=None,
    conda_env=None,
):

    if os.path.exists(path):
        raise ClearboxWrapperException(
            message="Model path '{}' already exists, model can not be saved.".format(
                path
            )
        )
        os.makedirs(path)


class Wrapper(GenericModel):
    pass
