from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from clearbox_wrapper.schema import _infer_schema, Schema

InferableDataset = Union[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]


class Signature(object):
    """Description of a Model, Preprocessing or Data Preparation inputs and outpus.

    Attributes
    ----------
    inputs : clearbox_wrapper.schema.Schema
        Inputs schema as a sequence of (optionally) named columns with types.
    outputs : Optional[clearbox_wrapper.schema.Schema]
        Outputs schema as a sequence of (optionally) named columns with types.

    """

    def __init__(self, inputs: Schema, outputs: Optional[Schema] = None) -> "Signature":
        """Create a new Signature instance given inputs and (optionally) outputs schemas.

        Parameters
        ----------
        inputs : clearbox_wrapper.schema.Schema
            Inputs schema as a sequence of (optionally) named columns with types.
        outputs : Optional[clearbox_wrapper.schema.Schema]
            Outputs schema as a sequence of (optionally) named columns with types,
            by default None.


        Raises
        ------
        TypeError
            If `inputs` is not type Schema or if `outputs` is not type Schema or None.
        """
        if not isinstance(inputs, Schema):
            raise TypeError("Inputs must be type Schema, got '{}'".format(type(inputs)))
        if outputs is not None and not isinstance(outputs, Schema):
            raise TypeError(
                "Outputs must be either None or clearbox_wrapper.schema.Schema, "
                "got '{}'".format(type(inputs))
            )
        self.inputs = inputs
        self.outputs = outputs

    def to_dict(self) -> Dict[str, Any]:
        """Generate dictionary representation of the signature.

        Returns
        -------
        Dict[str, Any]
            Signature dictionary {"inputs": inputs schema as JSON,
            "outputs": outputs schema as JSON}
        """
        return {
            "inputs": self.inputs.to_json(),
            "outputs": self.outputs.to_json() if self.outputs is not None else None,
        }

    @classmethod
    def from_dict(cls, signature_dict: Dict[str, Any]) -> "Signature":
        """Create a Signature instance from a dictionary representation.

        Parameters
        ----------
        signature_dict: Dict[str, Any]
            Signature dictionary {"inputs": inputs schema as JSON,
            "outputs": outputs schema as JSON}

        Returns
        -------
        Signature
            A Signature instance.
        """
        inputs = Schema.from_json(signature_dict["inputs"])
        if "outputs" in signature_dict and signature_dict["outputs"] is not None:
            outputs = Schema.from_json(signature_dict["outputs"])
            return cls(inputs, outputs)
        else:
            return cls(inputs)

    def __eq__(self, other: "Signature") -> bool:
        """Check if two Signature instances (self and other) are equal.

        Parameters
        ----------
        other : Signature
            A Signature instance

        Returns
        -------
        bool
            True if the two signatures are equal, False otherwise.
        """
        return (
            isinstance(other, Signature)
            and self.inputs == other.inputs
            and self.outputs == other.outputs
        )

    def __repr__(self) -> str:
        """Generate string representation.

        Returns
        -------
        str
            Signature string representation.
        """
        return (
            "inputs: \n"
            "  {}\n"
            "outputs: \n"
            "  {}\n".format(repr(self.inputs), repr(self.outputs))
        )


def infer_signature(input_data: Any, output_data: InferableDataset = None) -> Signature:
    """Infer a Signature from input data and (optionally) output data.

    The signature represents inputs and outputs scheme as  a sequence
    of (optionally) named columns with types.

    Parameters
    ----------
    input_data : Any
        Valid input data. E.g. (a subset of) the training dataset. It should be
        one of the following types:
        - pandas.DataFrame
        - dictionary of { name -> numpy.ndarray}
        - numpy.ndarray
        The element types should be mappable to one of `clearbox.schema.DataType`.
    output_data : InferableDataset, optional
        Valid output data. E.g. Preprocessed data or model predictions for
        (a subset of) the training dataset. It should be one of the following types:
        - pandas.DataFrame
        - numpy.ndarray
        The element types should be mappable to one of `clearbox.schema.DataType`.
        By default None.

    Returns
    -------
    Signature
        Inferred Signature
    """
    inputs = _infer_schema(input_data)
    outputs = _infer_schema(output_data) if output_data is not None else None
    return Signature(inputs, outputs)
