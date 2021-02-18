from datetime import datetime
import json
import os
from typing import Dict, Optional

import yaml

from clearbox_wrapper.schema import Schema
from clearbox_wrapper.signature import Signature


MLMODEL_FILE_NAME = "MLmodel"


class Model(object):
    """A ML Model that can support multiple model flavors."""

    def __init__(
        self,
        timestamp: Optional[datetime] = None,
        flavors: Optional[Dict] = None,
        model_signature: Optional[Signature] = None,
        preprocessing_signature: Optional[Signature] = None,
        preparation_signature: Optional[Signature] = None,
    ) -> None:
        """Create a new Model object.

        Parameters
        ----------
        timestamp : Optional[datetime], optional
            A timestamp of the model creation, by default None.
            If None, it will be use datetime.utcnow()
        flavors : Optional[Dict], optional
            Dictionary of flavors: a "flavor" is a convention
            representing the framework the model was created with
            and/or downstream tools that can "understand" the model,
            by default None.
        signature : Optional[Signature], optional
            Description of the model inputs as Signature oject,
            by default None
        """
        self.timestamp = str(timestamp or datetime.utcnow())
        self.flavors = flavors if flavors is not None else {}
        self.model_signature = model_signature
        self.preprocessing_signature = preprocessing_signature
        self.preparation_signature = preparation_signature

    def __eq__(self, other: "Model") -> bool:
        """Check if two models are equal.

        Parameters
        ----------
        other : Model
            A Model object

        Returns
        -------
        bool
            true, if the other Model is equal to self.
        """
        if not isinstance(other, Model):
            return False
        return self.__dict__ == other.__dict__

    def get_data_preparation_input_schema(self) -> Optional[Schema]:
        """Get the data preparation inputs schema."

        Returns
        -------
        Schema
            Data preparation inputs schema: specification of types and column names.
        """
        return (
            self.preparation_signature.inputs
            if self.preparation_signature is not None
            else None
        )

    def get_data_preparation_output_schema(self) -> Optional[Schema]:
        """Get the data preparation outputs schema."

        Returns
        -------
        Schema
            Data preparation outputs schema: specification of types and column names.
        """
        return (
            self.preparation_signature.outputs
            if self.preparation_signature is not None
            else None
        )

    def get_preprocessing_input_schema(self) -> Optional[Schema]:
        """Get the preprocessing inputs schema."

        Returns
        -------
        Schema
            Preprocessing inputs schema: specification of types and column names.
        """
        return (
            self.preprocessing_signature.inputs
            if self.preprocessing_signature is not None
            else None
        )

    def get_preprocessing_output_schema(self) -> Optional[Schema]:
        """Get the preprocessing outputs schema."

        Returns
        -------
        Schema
            Preprocessing outputs schema: specification of types and column names.
        """
        return (
            self.preprocessing_signature.outputs
            if self.preprocessing_signature is not None
            else None
        )

    def get_model_input_schema(self) -> Optional[Schema]:
        """Get the model inputs schema."

        Returns
        -------
        Schema
            Model inputs schema: specification of types and column names.
        """
        return self.model_signature.inputs if self.model_signature is not None else None

    def get_model_output_schema(self) -> Optional[Schema]:
        """Get the model outputs schema."

        Returns
        -------
        Schema
            Model outputs schema: specification of types and column names.
        """
        return (
            self.model_signature.outputs if self.model_signature is not None else None
        )

    def add_flavor(self, name: str, **params) -> "Model":
        """Add an entry for how to serve the model in a given format.

        Returns
        -------
        Model
            self
        """
        self.flavors[name] = params
        return self

    @property
    def preparation_signature(self) -> Optional[Signature]:
        """Get the data preparation signature associated with the model.

        Returns
        -------
        Optional[Signature]
            Data preparation signature as a Signature instance.
        """
        return self._preparation_signature

    @preparation_signature.setter
    def preparation_signature(self, value: Signature) -> None:
        """Set the data preparation signature of the model

        Parameters
        ----------
        value : Signature
            Data preparation signature as a Signature instance.
        """
        self._preparation_signature = value

    @property
    def preprocessing_signature(self) -> Optional[Signature]:
        """Get the preprocessing signature associated with the model.

        Returns
        -------
        Optional[Signature]
            Preprocessing signature as a Signature instance.
        """
        return self._preprocessing_signature

    @preprocessing_signature.setter
    def preprocessing_signature(self, value: Signature) -> None:
        """Set the preprocessing signature of the model

        Parameters
        ----------
        value : Signature
            Preprocessing signature as a Signature instance.
        """
        self._preprocessing_signature = value

    @property
    def model_signature(self) -> Optional[Signature]:
        """Get the model signature.

        Returns
        -------
        Optional[Signature]
            The model signature: it defines the schema of a model inputs and outputs.
            Model inputs/outputs are described as a sequence of (optionally) named
            columns with type.
        """
        return self._model_signature

    @model_signature.setter
    def model_signature(self, value: Signature) -> None:
        """Set the model signature

        Parameters
        ----------
        value : Signature
            Model signature as a Signature object.
        """
        self._model_signature = value

    def to_dict(self) -> Dict:
        """Get the model attributes as a dictionary.

        Returns
        -------
        Dict
            Model attributes as dict <attribute_name: attribute_value>
        """
        model_dict = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        if self.model_signature is not None:
            model_dict["signature"] = self.model_signature.to_dict()
        if self.preprocessing_signature is not None:
            model_dict[
                "preprocessing_signature"
            ] = self.preprocessing_signature.to_dict()
        if self.preparation_signature is not None:
            model_dict["preparation_signature"] = self.preparation_signature.to_dict()
        return model_dict

    def to_yaml(self, stream=None):
        """Serialize model object into a YAML stream. If stream is None,
        return the produced string instead.

        Parameters
        ----------
        stream : f, optional
            YAML stream, by default None

        Returns
        -------
        str
            YAML stream. If stream is None, return the produced string instead.
        """
        return yaml.safe_dump(self.to_dict(), stream=stream, default_flow_style=False)

    def __str__(self) -> str:
        """Get model representation as string.

        Returns
        -------
        str
            Model attributes as string.
        """
        return self.to_yaml()

    def to_json(self) -> str:
        """Get model representation in JSON format.

        Returns
        -------
        str
            Model attributes in JSON format.
        """
        return json.dumps(self.to_dict())

    def save(self, path: str) -> None:
        """Save model YAML representation to a file

        Parameters
        ----------
        path : str
            Path of the file to save the YAML in.
        """
        with open(path, "w") as out:
            self.to_yaml(out)

    @classmethod
    def load(cls, path: str) -> "Model":
        """Load a model from file.

        Parameters
        ----------
        path : str
            Path of the file to load the Model from.

        Returns
        -------
        Model
            Loaded Model.
        """
        if os.path.isdir(path):
            path = os.path.join(path, MLMODEL_FILE_NAME)
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f.read()))

    @classmethod
    def from_dict(cls, model_dict: Dict) -> "Model":
        """Load a model from its YAML representation.

        Parameters
        ----------
        model_dict : Dict
            Model dictionary representation.

        Returns
        -------
        Model
            Loaded Model.
        """

        if "signature" in model_dict and isinstance(model_dict["signature"], dict):
            model_dict = model_dict.copy()
            model_dict["model_signature"] = Signature.from_dict(model_dict["signature"])
            del model_dict["signature"]

        if "preprocessing_signature" in model_dict and isinstance(
            model_dict["preprocessing_signature"], dict
        ):
            model_dict = model_dict.copy()
            model_dict["preprocessing_signature"] = Signature.from_dict(
                model_dict["preprocessing_signature"]
            )

        if "preparation_signature" in model_dict and isinstance(
            model_dict["preparation_signature"], dict
        ):
            model_dict = model_dict.copy()
            model_dict["preparation_signature"] = Signature.from_dict(
                model_dict["preparation_signature"]
            )

        return cls(**model_dict)
