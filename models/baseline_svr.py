""" Implementation of baseline SVR model using BoW features """
import utils
import pickle
from models import base_model
from sklearn.svm import SVR
import numpy as np


class Model(base_model.Model):
    """ Implementation of SVR Baseline Model """

    def __init__(self, X_train, y_train, X_test, y_test, optimization_parameters={}):
        base_model.Model.__init__(self, X_train, y_train, X_test, y_test, optimization_parameters)

    def train(self):
        """ Train the SVR Baseline Model on all emotions """
        self.train_svr_models = {}
        for emotion, features in self.X_train.iteritems():
            print("Training on {0} dataset".format(emotion))
            svr = SVR() # use the default hyperparameters
            self.train_svr_models[emotion] = svr.fit(features, self.y_train[emotion])

    def optimize(self):
        """ Optimize the SVR Baseline Model. We don't want to optimize our baseline model"""
        return None

    def predict(self):
        """ Predict on new test data """
        predictions = {}
        for emotion, features in self.X_test.iteritems():
            print("Predicting on {0}".format(emotion))
            model = self.train_svr_models[emotion] 
            emotion_predictions = model.predict(features)
            predictions[emotion] = emotion_predictions
        return predictions

    def evaluate(self, predictions, ground_truth, metric):
        eval_metrics = {}
        if metric == "rmse":
            for emotion, predicts in predictions.iteritems():
                ground_truth_predictions = ground_truth[emotion]
                val = utils.metrics.rmse(ground_truth_predictions, predicts)
                eval_metrics[emotion] = val
        elif metric == "mse":
            for emotion, predicts in predictions.iteritems():
                ground_truth_predictions = ground_truth[emotion]
                val = utils.metrics.mse(ground_truth_predictions, predicts)
                eval_metrics[emotion] = val
        elif metric == "pearson":
            for emotion, predicts in predictions.iteritems():
                ground_truth_predictions = ground_truth[emotion]
                val = utils.metrics.pearson_correlation(ground_truth_predictions, predicts)
                eval_metrics[emotion] = val
        else:
            return 0
        avg_metric = np.mean(eval_metrics.values())
        return eval_metrics, avg_metric

    def save(self, save_location):
        with open(save_location, "w") as f:
            pickle.dump(self.model, f)
