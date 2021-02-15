from enum import Enum
import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from clearbox_wrapper.exceptions import ClearboxWrapperException


def _pandas_string_type():
    try:
        return pd.StringDtype()
    except AttributeError:
        return np.object


class DataType(Enum):

    boolean = (1, np.dtype("bool"), "BooleanType")  # Logical data (True, False)
    integer = (2, np.dtype("int32"), "IntegerType")  # 32b signed integer numbers
    long = (3, np.dtype("int64"), "LongType")  # 64b signed integer numbers
    float = (4, np.dtype("float32"), "FloatType")  # 32b floating point numbers
    double = (5, np.dtype("float64"), "DoubleType")  # 64b floating point numbers
    string = (6, np.dtype("str"), "StringType", _pandas_string_type())  # Text data
    binary = (7, np.dtype("bytes"), "BinaryType", np.object)  # Sequence of raw bytes


class ColumnSpec(object):
    def __init__(
        self,
        type: DataType,
        name: Optional[str] = None,
    ):
        self._name = name
        try:
            self._type = DataType[type] if isinstance(type, str) else type
        except KeyError:
            raise ClearboxWrapperException(
                "Unsupported type '{0}', expected instance of DataType or "
                "one of {1}".format(type, [t.name for t in DataType])
            )
        if not isinstance(self.type, DataType):
            raise TypeError(
                "Expected Datatype or str for the 'type' "
                "argument, but got {}".format(self.type.__class__)
            )

    @property
    def type(self) -> DataType:
        """The column data type."""
        return self._type

    @property
    def name(self) -> Optional[str]:
        """The column name or None if the columns is unnamed."""
        return self._name

    def to_dict(self) -> Dict[str, Any]:
        if self.name is None:
            return {"type": self.type.name}
        else:
            return {"name": self.name, "type": self.type.name}

    def __eq__(self, other) -> bool:
        names_eq = (self.name is None and other.name is None) or self.name == other.name
        return names_eq and self.type == other.type

    def __repr__(self) -> str:
        if self.name is None:
            return repr(self.type)
        else:
            return "{name}: {type}".format(name=repr(self.name), type=repr(self.type))


class Schema(object):
    """
    Specification of types and column names in a dataset.
    Schema is represented as a list of :py:class:`ColumnSpec`.
    The columns in a schema can be named, with unique non empty name for every column,
    or unnamed with implicit integer index defined by their list indices.
    Combination of named and unnamed columns is not allowed.
    """

    def __init__(self, cols: List[ColumnSpec]):
        if not (
            all(map(lambda x: x.name is None, cols))
            or all(map(lambda x: x.name is not None, cols))
        ):
            raise ClearboxWrapperException(
                "Creating Schema with a combination of named and unnamed columns "
                "is not allowed. Got column names {}".format([x.name for x in cols])
            )
        self._cols = cols

    @property
    def columns(self) -> List[ColumnSpec]:
        """The list of columns that defines this schema."""
        return self._cols

    def column_names(self) -> List[Union[str, int]]:
        """Get list of column names or range of indices if the schema has no column names."""
        return [x.name or i for i, x in enumerate(self.columns)]

    def has_column_names(self) -> bool:
        """ Return true iff this schema declares column names, false otherwise. """
        return self.columns and self.columns[0].name is not None

    def column_types(self) -> List[DataType]:
        """ Get column types of the columns in the dataset."""
        return [x.type for x in self._cols]

    def numpy_types(self) -> List[np.dtype]:
        """ Convenience shortcut to get the datatypes as numpy types."""
        return [x.type.to_numpy() for x in self.columns]

    def pandas_types(self) -> List[np.dtype]:
        """ Convenience shortcut to get the datatypes as pandas types."""
        return [x.type.to_pandas() for x in self.columns]

    def to_json(self) -> str:
        """Serialize into json string."""
        return json.dumps([x.to_dict() for x in self.columns])

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialize into a jsonable dictionary."""
        return [x.to_dict() for x in self.columns]

    @classmethod
    def from_json(cls, json_str: str):
        """ Deserialize from a json string."""
        return cls([ColumnSpec(**x) for x in json.loads(json_str)])

    def __eq__(self, other) -> bool:
        if isinstance(other, Schema):
            return self.columns == other.columns
        else:
            return False

    def __repr__(self) -> str:
        return repr(self.columns)
