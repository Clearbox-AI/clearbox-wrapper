from typing import Any, Dict, Union

import numpy as np
import pandas as pd

from clearbox_wrapper.schema import _infer_schema, Schema

InferableDataset = Union[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]


class Signature(object):
    def __init__(self, inputs: Schema, outputs: Schema = None):
        if not isinstance(inputs, Schema):
            raise TypeError("inputs must be type Schema, got '{}'".format(type(inputs)))
        if outputs is not None and not isinstance(outputs, Schema):
            raise TypeError(
                "outputs must be either None or mlflow.models.signature.Schema, "
                "got '{}'".format(type(inputs))
            )
        self.inputs = inputs
        self.outputs = outputs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inputs": self.inputs.to_json(),
            "outputs": self.outputs.to_json() if self.outputs is not None else None,
        }

    @classmethod
    def from_dict(cls, signature_dict: Dict[str, Any]):
        """
        Deserialize from dictionary representation.
        :param signature_dict: Dictionary representation of model signature.
                               Expected dictionary format:
                               `{'inputs': <json string>}`
        :return: Signature populated with the data form the dictionary.
        """
        inputs = Schema.from_json(signature_dict["inputs"])
        if "outputs" in signature_dict and signature_dict["outputs"] is not None:
            outputs = Schema.from_json(signature_dict["outputs"])
            return cls(inputs, outputs)
        else:
            return cls(inputs)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Signature)
            and self.inputs == other.inputs
            and self.outputs == other.outputs
        )

    def __repr__(self) -> str:
        return (
            "inputs: \n"
            "  {}\n"
            "outputs: \n"
            "  {}\n".format(repr(self.inputs), repr(self.outputs))
        )


def infer_signature(input_data: Any, output_data: InferableDataset = None) -> Signature:
    """
    Infer an MLflow model signature from the training data (input).
    The signature represents model input as data frames with (optionally) named columns
    and data type specified as one of types defined in :py:class:`mlflow.types.DataType`.
    This method will raise an exception if the user data contains incompatible types or is not
    passed in one of the supported formats listed below.
    The input should be one of these:
      - pandas.DataFrame
      - dictionary of { name -> numpy.ndarray}
      - numpy.ndarray
      - pyspark.sql.DataFrame
    The element types should be mappable to one of :py:class:`mlflow.types.DataType`.
    NOTE: Multidimensional (>2d) arrays (aka tensors) are not supported at this time.
    :param model_input: Valid input to the model. E.g. (a subset of) the training dataset.
    :param model_output: Valid model output. E.g. Model predictions for the (subset of) training
                         dataset.
    :return: Signature
    """
    inputs = _infer_schema(input_data)
    outputs = _infer_schema(output_data) if output_data is not None else None
    return Signature(inputs, outputs)
