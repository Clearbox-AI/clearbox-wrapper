import mlflow.pyfunc

from typing import Any, Optional


class ClearboxWrapper(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        model: Any,
        preprocessing: Optional[Any] = None,
        additional_preprocessing: Optional[Any] = None,
    ) -> "ClearboxWrapper":
        if preprocessing is None and additional_preprocessing is not None:
            raise ValueError(
                "Attribute 'preprocessing' is None but attribute "
                "'additional_preprocessing' is not None. If you have a single step "
                "preprocessing, pass it as attribute 'preprocessing'"
            )
        self.model = model
        self.preprocessing = preprocessing
        self.additional_preprocessing = additional_preprocessing

    def predict(self, context=None, model_input=None):
        if self.preprocessing is not None:
            model_input = (
                self.preprocessing.transform(model_input)
                if "transform" in dir(self.preprocessing)
                else self.preprocessing(model_input)
            )
        if self.additional_preprocessing is not None:
            model_input = (
                self.additional_preprocessing.transform(model_input)
                if "transform" in dir(self.additional_preprocessing)
                else self.additional_preprocessing(model_input)
            )
        return self.model.predict_proba(model_input)

    def save(self, path: str) -> None:
        mlflow.set_tracking_uri(path)
        mlflow.pyfunc.save_model(path=path, python_model=self)
