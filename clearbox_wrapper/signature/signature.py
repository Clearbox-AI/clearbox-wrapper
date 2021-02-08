from typing import Any, Dict

from clearbox_wrapper.signature.schema import Schema
from clearbox_wrapper.signature.utils import _infer_schema


class ModelSignature(object):
    def __init__(self, inputs: Schema):
        if not isinstance(inputs, Schema):
            raise TypeError("inputs must be type Schema, got '{}'".format(type(inputs)))
        self.inputs = inputs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inputs": self.inputs.to_json(),
        }

    @classmethod
    def from_dict(cls, signature_dict: Dict[str, Any]):
        """
        Deserialize from dictionary representation.
        :param signature_dict: Dictionary representation of model signature.
                               Expected dictionary format:
                               `{'inputs': <json string>}`
        :return: ModelSignature populated with the data form the dictionary.
        """
        inputs = Schema.from_json(signature_dict["inputs"])
        return cls(inputs)

    def __eq__(self, other) -> bool:
        return isinstance(other, ModelSignature) and self.inputs == other.inputs

    def __repr__(self) -> str:
        return "inputs: \n" "  {}\n".format(repr(self.inputs))


def infer_signature(model_input: Any) -> ModelSignature:
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
    :return: ModelSignature
    """
    inputs = _infer_schema(model_input)
    return ModelSignature(inputs)
