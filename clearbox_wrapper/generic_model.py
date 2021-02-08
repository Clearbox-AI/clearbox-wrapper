from abc import ABCMeta, abstractmethod


class Model(object, metaclass=ABCMeta):
    @abstractmethod
    def prepare_data(self, input_data):
        pass

    @abstractmethod
    def preprocess_data(self, input_data):
        pass

    @abstractmethod
    def predit(self, context, input_data):
        pass
