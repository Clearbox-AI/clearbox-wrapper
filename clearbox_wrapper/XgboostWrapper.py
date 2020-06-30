import pickle

from .ClearBoxWrapper import ClearBoxWrapper


class XgboostWrapper(ClearBoxWrapper):
    def __init__(self, model):
        super().__init__()

        if type(model) is str:
            with open(model, "rb") as model_file:
                self.model = pickle.load(model_file)
        else:
            self.model = model
