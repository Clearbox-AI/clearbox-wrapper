import os
import cloudpickle

from .ClearBoxWrapper import ClearBoxWrapper


class KerasWrapper(ClearBoxWrapper):
    def __init__(self, model):
        super().__init__()

        if type(model) is str:
            from tensorflow import keras
            self.model = keras.models.load_model(model)
            self.model_path = os.path.split(model)[1]
        else:
            self.model = model
            self.model_path = './keras_model'

    def dump(self, path):
        self.model.save(self.model_path)
        self.model = None
        cloudpickle.dump(self, open(path, 'wb'))

    def dumps(self):
        self.model.save(self.model_path)
        self.model = None
        return cloudpickle.dumps(self)

    @staticmethod
    def load(path):
        from tensorflow import keras
        clearbox_wrapper = cloudpickle.load(open(path, 'rb'))
        print(os.path.split(path)[0] + clearbox_wrapper.model_path)
        clearbox_wrapper.model = keras.models.load_model(os.path.split(path)[0] + clearbox_wrapper.model_path)
        return clearbox_wrapper

    @staticmethod
    def loads(wrapper):
        from tensorflow import keras
        clearbox_wrapper = cloudpickle.loads(wrapper)
        clearbox_wrapper.model = keras.models.load_model(clearbox_wrapper.model_path)
        return clearbox_wrapper