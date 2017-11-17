import abc

class Model(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, X_train, y_train, X_test, y_test, optimization_parameters={}):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def optimize(self):
        pass
        
    @abc.abstractmethod
    def predict(self):
        pass

    @abc.abstractmethod
    def evaluate(self, metric):
        pass

    @abc.abstractmethod
    def save(self, filename):
        pass


