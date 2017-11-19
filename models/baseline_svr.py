""" Implementation of baseline SVR model using BoW features """
import utils
import pickle
from models import base_model
from sklearn.svm import SVR


class Model(base_model.Model):
    """ Implementation of SVR Baseline Model """

    def __init__(self, X_train, y_train, X_test, y_test, optimization_parameters={}):
        base_model.Model.__init__(self, X_train, y_train, X_test, y_test, optimization_parameters)

    def train(self):
        """ Train the SVR Baseline Model """
        print("Training SVR Baseline Model")
        svr = SVR()  # use default hyperparameters
        svr = svr.fit(self.X_train, self.y_train)
        self.model = svr

    def optimize(self):
        """ Optimize the SVR Baseline Model. We don't want to optimize our baseline model"""
        return None

    def predict(self):
        """ Predict on new test data """
        return self.model.predict(self.X_test)

    def evaluate(self, metric):
        predictions = self.predict()
        print "Evaluating on {0}".format(metric)

        if metric == "rmse":
            return utils.metrics.rmse(self.y_test, predictions)
        elif metric == "mse":
            return utils.metrics.mse(self.y_test, predictions)
        else:
            return 0

    def save(self, save_location):
        with open(save_location, "w") as f:
            pickle.dump(self.model, f)
