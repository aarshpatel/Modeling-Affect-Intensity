import abc


class Model(object):
    """ Represents an abstract base class """
    __metaclass__ = abc.ABCMeta

    def __init__(self, X_train, y_train, optimization_parameters={}):
        self.X_train = X_train
        self.y_train = y_train
        self.optimization_parameters = optimization_parameters
        self.model = None

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def optimize(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass

    def score(self):
        if self.model is not None:
            return self.model.score(self.X_test, self.y_test)

    @abc.abstractmethod
    def save(self, filename):
        pass
