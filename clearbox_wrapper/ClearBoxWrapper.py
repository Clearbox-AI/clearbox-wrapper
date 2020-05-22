import cloudpickle


class ClearBoxWrapper(object):
    def __init__(self):
        pass

    def transform(self):
        pass

    def predict(self):
        pass

    def dump(self, path):
        with open(path, 'wb') as wrapper_file:
            cloudpickle.dump(self, wrapper_file)

    def dumps(self):
        return cloudpickle.dumps(self)