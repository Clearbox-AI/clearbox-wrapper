from typing import Any

import numpy as np
import pandas as pd

from clearbox_wrapper.exceptions import ClearboxWrapperException
from .schema import ColumnSpec, DataType, Schema


class TensorsNotSupportedException(ClearboxWrapperException):
    def __init__(self, msg):
        super().__init__(
            "Multidimensional arrays (aka tensors) are not supported. " "{}".format(msg)
        )


def _infer_schema(data: Any) -> Schema:
    """Infer a schema from data.

    The schema represents data as a sequence of (optionally) named columns with types.

    Parameters
    ----------
    data : Any
        Valid data. It should be one of the following types:
        - pandas.DataFrame or pandas.Series
        - dictionary of { name -> numpy.ndarray}
        - numpy.ndarray
        The data types should be mappable to one of  `clearbox.schema.DataType`.

    Returns
    -------
    Schema
        Schema instance inferred from data.

    Raises
    ------
    TypeError
        If type of data is not valid (pandas.DataFrame or pandas.Series, dictionary of
        { name -> numpy.ndarray}, numpy.ndarray)
    TensorsNotSupportedException
        If data are multidimensional (>2d) arrays (tensors).
    """
    if hasattr(data, "toarray"):
        data = data.toarray()
    if isinstance(data, dict):
        res = []
        for col in data.keys():
            ary = data[col]
            if not isinstance(ary, np.ndarray):
                raise TypeError("Data in the dictionary must be of type numpy.ndarray")
            dims = len(ary.shape)
            if dims == 1:
                res.append(ColumnSpec(type=_infer_numpy_array(ary), name=col))
            else:
                raise TensorsNotSupportedException(
                    "Data in the dictionary must be 1-dimensional, "
                    "got shape {}".format(ary.shape)
                )
        schema = Schema(res)
    elif isinstance(data, pd.Series):
        has_nans = data.isna().any()
        series_converted_to_numpy = data.dropna().values if has_nans else data.values
        schema = Schema(
            [
                ColumnSpec(
                    type=_infer_numpy_array(series_converted_to_numpy),
                    has_nans=has_nans,
                )
            ]
        )
    elif isinstance(data, pd.DataFrame):
        columns_spec_list = []
        for col in data.columns:
            has_nans = data[col].isna().any()
            col_converted_to_numpy = (
                data[col].dropna().values if has_nans else data[col].values
            )
            columns_spec_list.append(
                ColumnSpec(
                    type=_infer_numpy_array(col_converted_to_numpy),
                    name=col,
                    has_nans=has_nans,
                )
            )
        schema = Schema(columns_spec_list)
    elif isinstance(data, np.ndarray):
        if len(data.shape) > 2:
            raise TensorsNotSupportedException(
                "Attempting to infer schema from numpy array with "
                "shape {}".format(data.shape)
            )
        if data.dtype == np.object:
            data = pd.DataFrame(data).infer_objects()
            schema = Schema(
                [
                    ColumnSpec(type=_infer_numpy_array(data[col].values))
                    for col in data.columns
                ]
            )
        elif len(data.shape) == 1:
            schema = Schema([ColumnSpec(type=_infer_numpy_dtype(data.dtype))])
        elif len(data.shape) == 2:
            schema = Schema(
                [ColumnSpec(type=_infer_numpy_dtype(data.dtype))] * data.shape[1]
            )
    else:
        raise TypeError(
            "Expected one of (pandas.DataFrame, numpy array, "
            "dictionary of (name -> numpy.ndarray)) "
            "but got '{}'".format(type(data))
        )
    return schema


def _infer_numpy_dtype(dtype: np.dtype) -> DataType:
    """Infer DataType from numpy dtype.

    Parameters
    ----------
    dtype : np.dtype
        Numpy dtype

    Returns
    -------
    DataType
        Inferred DataType.

    Raises
    ------
    TypeError
        If type of `dtype` is not numpy.dtype.
    Exception
        If `dtype.kind`=='O'
    ClearboxWrapperException
        If `dtype` is unsupported.
    """
    if not isinstance(dtype, np.dtype):
        raise TypeError("Expected numpy.dtype, got '{}'.".format(type(dtype)))
    if dtype.kind == "b":
        return DataType.boolean
    elif dtype.kind == "i" or dtype.kind == "u":
        if dtype.itemsize < 4 or (dtype.kind == "i" and dtype.itemsize == 4):
            return DataType.integer
        elif dtype.itemsize < 8 or (dtype.kind == "i" and dtype.itemsize == 8):
            return DataType.long
    elif dtype.kind == "f":
        if dtype.itemsize <= 4:
            return DataType.float
        elif dtype.itemsize <= 8:
            return DataType.double

    elif dtype.kind == "U":
        return DataType.string
    elif dtype.kind == "S":
        return DataType.binary
    elif dtype.kind == "O":
        raise Exception(
            "Can not infer np.object without looking at the values, call "
            "_infer_numpy_array instead."
        )
    raise ClearboxWrapperException(
        "Unsupported numpy data type '{0}', kind '{1}'".format(dtype, dtype.kind)
    )


def _infer_numpy_array(col: np.ndarray) -> DataType:
    """Infer DataType of a numpy array.

    Parameters
    ----------
    col : np.ndarray
        Column representation as a numpy array.

    Returns
    -------
    DataType
        Inferred datatype.

    Raises
    ------
    TypeError
        If `col` is not a numpy array.
    ClearboxWrapperException
        If `col` is not a 1D array.
    """
    if not isinstance(col, np.ndarray):
        raise TypeError("Expected numpy.ndarray, got '{}'.".format(type(col)))
    if len(col.shape) > 1:
        raise ClearboxWrapperException(
            "Expected 1d array, got array with shape {}".format(col.shape)
        )

    class IsInstanceOrNone(object):
        def __init__(self, *args):
            self.classes = args
            self.seen_instances = 0

        def __call__(self, x):
            if x is None:
                return True
            elif any(map(lambda c: isinstance(x, c), self.classes)):
                self.seen_instances += 1
                return True
            else:
                return False

    if col.dtype.kind == "O":
        is_binary_test = IsInstanceOrNone(bytes, bytearray)
        if all(map(is_binary_test, col)) and is_binary_test.seen_instances > 0:
            return DataType.binary
        is_string_test = IsInstanceOrNone(str)
        if all(map(is_string_test, col)) and is_string_test.seen_instances > 0:
            return DataType.string
        # NB: bool is also instance of int => boolean test must precede integer test.
        is_boolean_test = IsInstanceOrNone(bool)
        if all(map(is_boolean_test, col)) and is_boolean_test.seen_instances > 0:
            return DataType.boolean
        is_long_test = IsInstanceOrNone(int)
        if all(map(is_long_test, col)) and is_long_test.seen_instances > 0:
            return DataType.long
        is_double_test = IsInstanceOrNone(float)
        if all(map(is_double_test, col)) and is_double_test.seen_instances > 0:
            return DataType.double
        else:
            raise ClearboxWrapperException(
                "Unable to map 'np.object' type to MLflow DataType. np.object can"
                "be mapped iff all values have identical data type which is one "
                "of (string, (bytes or byterray),  int, float)."
            )
    else:
        return _infer_numpy_dtype(col.dtype)
