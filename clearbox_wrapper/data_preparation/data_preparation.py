import os
from typing import Callable, Union

import dill
import numpy as np
import pandas as pd

from clearbox_wrapper.exceptions import ClearboxWrapperException

dill.settings["recurse"] = True
DataPreparationInput = Union[pd.DataFrame, pd.Series, np.ndarray]
DataPreparationOutput = Union[pd.DataFrame, pd.Series, np.ndarray]


class DataPreparation(object):
    def __init__(self, data_preparation_function: Callable):
        """Create a DataPreparation instance.

        Parameters
        ----------
        preprocessing_function : Callable
            A function to use as a preprocessor. You can use your own custom code for
            preprocessing, but it must be wrapped in a single function.

            NOTE: If the preprocessing includes any kind of fitting on the training dataset
            (e.g. Scikit Learn transformers), it must be performed outside the final
            preprocessing function to save. Fit the transformer(s) outside the function and
            put only the transform method inside it. Furthermore, if the entire preprocessing
            is performed with a single Scikit-Learn transformer, you can directly pass it
            (fitted) to this method.

        Raises
        ------
        TypeError
            If preprocessing_function is not a function (Callable type)
        """
        self.data_preparation = data_preparation_function

    def __repr__(self) -> str:
        return "Data Preparation: \n" "  {}\n".format(repr(self.data_preparation))

    @property
    def data_preparation(self) -> Callable:
        """Get the data preparation function.

        Returns
        -------
        Callable
            The data preparation function.
        """
        return self._data_preparation

    @data_preparation.setter
    def data_preparation(self, data_preparation_function: Callable) -> None:
        """Set the data preparation function.

        Parameters
        ----------
        value : Callable
            The data preparation function.
        """
        self._data_preparation = data_preparation_function

    def prepare_data(self, data: DataPreparationInput) -> DataPreparationOutput:
        """Prepare input data using the data preparation function.

        Parameters
        ----------
        data : DataPreparationInput
            Input data to prepare.

        Returns
        -------
        DataPreparationOutput
            Prepared data.
        """
        prepared_data = (
            self.data_preparation.transform(data)
            if hasattr(self.data_preparation, "transform")
            else self.data_preparation(data)
        )
        return prepared_data

    def save(self, path: str) -> None:
        if os.path.exists(path):
            raise ClearboxWrapperException(
                "Data preparation path '{}' already exists".format(path)
            )
        with open(path, "wb") as data_preparation_serialized_file:
            dill.dump(self, data_preparation_serialized_file)


def create_and_save_data_preparation(
    data_preparation_function: Callable, path: str
) -> None:
    """Create, serialize and save a DataPreparation instance.

    Parameters
    ----------
    data_preparation_function : Callable
        A function to use as data preparation. You can use your own custom code for
        data preparation, but it must be wrapped in a single function.

        NOTE: If the data preparation includes any kind of fitting on the training dataset
        (e.g. Scikit Learn transformers), it must be performed outside the final data
        preparation function to save. Fit the transformer(s) outside the function and put
        only the transform method inside it. Furthermore, if the entire data preparation
        is performed with a single Scikit-Learn transformer, you can directly pass it
        (fitted) to this method.
    path : str
        Local path to save the data preparation to.

    Raises
        ------
        TypeError
            If data_preparation_function is not a function (Callable type)
        ClearboxWrapperException
            If data preparation path already exists.
    """
    if not isinstance(data_preparation_function, Callable):
        raise TypeError(
            "data_preparation_function should be a Callable, got '{}'".format(
                type(data_preparation_function)
            )
        )
    if os.path.exists(path):
        raise ClearboxWrapperException(
            "Data preparation path '{}' already exists".format(path)
        )

    data_preparation = DataPreparation(data_preparation_function)
    with open(path, "wb") as data_preparation_serialized_file:
        dill.dump(data_preparation, data_preparation_serialized_file)


def load_serialized_data_preparation(
    serialized_data_preparation_path: str,
) -> DataPreparation:
    with open(serialized_data_preparation_path, "rb") as serialized_data_preparation:
        return dill.load(serialized_data_preparation)
