import cloudpickle

from .ClearBoxWrapper import ClearBoxWrapper


class PytorchWrapper(ClearBoxWrapper):
    def __init__(self, model):
        super().__init__()

        if type(model) is str:
            self.model = cloudpickle.load(open(model, 'rb'))
        else:
            self.model = model
