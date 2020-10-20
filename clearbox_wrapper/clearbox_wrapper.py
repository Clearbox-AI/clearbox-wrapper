import mlflow.pyfunc

from typing import Any, Optional


class ClearboxWrapper(mlflow.pyfunc.PythonModel):
    def __init__(
        self, model: Any, preprocessing: Optional[Any] = None
    ) -> "ClearboxWrapper":
        self.model = model
        self.preprocessing = preprocessing

    def predict(self, context=None, model_input=None):
        if self.preprocessing is not None:
            if "transform" in dir(self.preprocessing):
                model_input = self.preprocessing.transform(model_input)
            else:
                model_input = self.preprocessing(model_input)
        return self.model.predict_proba(model_input)

    def save(self, path: str) -> None:
        mlflow.set_tracking_uri(path)
        mlflow.pyfunc.save_model(path=path, python_model=self)
