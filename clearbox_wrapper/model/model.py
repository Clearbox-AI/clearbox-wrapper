from datetime import datetime
import json
import os

import yaml


MLMODEL_FILE_NAME = "MLmodel"


class Model(object):
    def __init__(
        self,
        artifacts_path=None,
        timestamp=None,
        signature=None,
    ):
        self.timestamp = str(timestamp or datetime.utcnow())
        self.signature = signature

    def __eq__(self, other):
        if not isinstance(other, Model):
            return False
        return self.__dict__ == other.__dict__

    def get_input_schema(self):
        return self.signature.inputs if self.signature is not None else None

    def get_output_schema(self):
        return self.signature.outputs if self.signature is not None else None

    @property
    def signature(self):  # -> Optional[ModelSignature]
        return self.signature

    @signature.setter
    def signature(self, value):
        self.signature = value

    def to_dict(self):
        res = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        if self.signature is not None:
            res["signature"] = self.signature.to_dict()
        return res

    def to_yaml(self, stream=None):
        return yaml.safe_dump(self.to_dict(), stream=stream, default_flow_style=False)

    def __str__(self):
        return self.to_yaml()

    def to_json(self):
        return json.dumps(self.to_dict())

    def save(self, path):
        with open(path, "w") as out:
            self.to_yaml(out)

    @classmethod
    def load(cls, path):
        if os.path.isdir(path):
            path = os.path.join(path, MLMODEL_FILE_NAME)
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f.read()))

    @classmethod
    def from_dict(cls, model_dict):
        """Load a model from its YAML representation."""

        from clearbox_wrapper.signature import ModelSignature

        if "signature" in model_dict and isinstance(model_dict["signature"], dict):
            model_dict = model_dict.copy()
            model_dict["signature"] = ModelSignature.from_dict(model_dict["signature"])

        return cls(**model_dict)
