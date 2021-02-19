import os
from typing import Callable, Union

import dill
import numpy as np
import pandas as pd

from clearbox_wrapper.exceptions import ClearboxWrapperException


dill.settings["recurse"] = True
PreprocessingInput = Union[pd.DataFrame, pd.Series, np.ndarray]
PreprocessingOutput = Union[pd.DataFrame, pd.Series, np.ndarray]


class Preprocessing(object):
    def __init__(self, preprocessing_function: Callable):
        """Create a Preprocessing instance.

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
        self.preprocessing = preprocessing_function

    def __repr__(self) -> str:
        return "Preprocesing: \n" "  {}\n".format(repr(self.preprocessing))

    @property
    def preprocessing_function(self) -> Callable:
        """Get the preprocessing function.

        Returns
        -------
        Callable
            The preprocessing function.
        """
        return self._preprocessing

    @preprocessing_function.setter
    def preprocessing_function(self, preprocessing_function: Callable) -> None:
        """Set the preprocessing function.

        Parameters
        ----------
        value : Callable
            The preprocessing function.
        """
        self._preprocessing = preprocessing_function

    def preprocess(self, data: PreprocessingInput) -> PreprocessingOutput:
        """Preprocess input data using the preprocessing function.

        Parameters
        ----------
        data : PreprocessingInput
            Input data to preprocess.

        Returns
        -------
        PreprocessingOutput
            Preprocessed data.
        """
        preprocessed_data = (
            self.preprocessing.transform(data)
            if hasattr(self.preprocessing, "transform")
            else self.preprocessing(data)
        )
        return preprocessed_data

    def save(self, path: str) -> None:
        if os.path.exists(path):
            raise ClearboxWrapperException(
                "Preprocessing path '{}' already exists".format(path)
            )
        with open(path, "wb") as preprocessing_serialized_file:
            dill.dump(self, preprocessing_serialized_file)


def create_and_save_preprocessing(preprocessing_function: Callable, path: str) -> None:
    """Create, serialize and save a Preprocessing instance.

    Parameters
    ----------
    preprocessing_function : Callable
        A function to use as a preprocessor. You can use your own custom code for
        preprocessing, but it must be wrapped in a single function.

        NOTE: If the preprocessing includes any kind of fitting on the training dataset
        (e.g. Scikit Learn transformers), it must be performed outside the final preprocessing
        function to save. Fit the transformer(s) outside the function and put only the transform
        method inside it. Furthermore, if the entire preprocessing is performed with a single
        Scikit-Learn transformer, you can directly pass it (fitted) to this method.
    path : str
        Local path to save the preprocessing to.

    Raises
        ------
        TypeError
            If preprocessing_function is not a function (Callable type)
        ClearboxWrapperException
            If preprocessing path already exists.
    """
    if not isinstance(preprocessing_function, Callable):
        raise TypeError(
            "preprocessing_function should be a Callable, got '{}'".format(
                type(preprocessing_function)
            )
        )
    if os.path.exists(path):
        raise ClearboxWrapperException(
            "Preprocessing path '{}' already exists".format(path)
        )

    preprocessing = Preprocessing(preprocessing_function)
    with open(path, "wb") as preprocessing_serialized_file:
        dill.dump(preprocessing, preprocessing_serialized_file)


def load_serialized_preprocessing(serialized_preprocessing_path: str) -> Preprocessing:
    with open(serialized_preprocessing_path, "rb") as serialized_preprocessing:
        return dill.load(serialized_preprocessing)
