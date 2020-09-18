import cloudpickle


class ClearBoxWrapper(object):
    def __init__(self):
        pass

    def predict(self, X):
        pass

    def dump(self, path):
        cloudpickle.dump(self, open(path, 'wb'))

    def dumps(self):
        return cloudpickle.dumps(self)

    @staticmethod
    def load(path):
        model = cloudpickle.load(open(path, 'rb'))
        return model

    @staticmethod
    def loads(wrapper):
        model = cloudpickle.loads(wrapper)
        return model
