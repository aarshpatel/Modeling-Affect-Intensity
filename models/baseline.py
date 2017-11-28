""" Implementation of baseline model using any set of features """
import utils
import pickle
from models import base_model
import numpy as np
from sklearn.model_selection import GridSearchCV

class Model(base_model.Model):
    """ Implementation of baseline model """

    def __init__(self, X_train, y_train, X_test, y_test, optimization_parameters={}, sklearn_model=None):
        base_model.Model.__init__(self, X_train, y_train, X_test, y_test, optimization_parameters)
        self.sklearn_model = sklearn_model

    def train(self, optimization_parameters=None):
        """ Train the baseline model on all emotions """
        self.trained_models = {}
        for emotion, features in self.X_train.iteritems():
            model = self.sklearn_model() # use the default hyperparameters
            if optimization_parameters:
                model.set_params(**optimization_parameters[emotion])

            self.trained_models[emotion] = model.fit(features, self.y_train[emotion])

    def optimize(self, metric):
        """ 
        Optimize the baseline model. We don't want to optimize our baseline model
        
        @metric: the metric to optimize model
        """

        print('Optimizing model {0} using {1}'.format(repr(self.sklearn_model), metric))

        if metric == "rmse":
            scoring_metric = utils.metrics.rmse_scorer
        elif metric == "mse":
            scoring_metric = utils.metrics.mse_scorer
        elif metric == "pearson_correlation":
            scoring_metric = utils.metrics.pearson_scorer

        optimized_models = {}
        for emotion, features in self.X_train.iteritems():

            grid = GridSearchCV(self.sklearn_model(), self.optimization_parameters, n_jobs=-1, cv=5, scoring=scoring_metric)
            grid.fit(features, self.y_train[emotion])

            optimized_models[emotion] = grid.best_params_
        return optimized_models

    def predict(self):
        """ Predict on new test data """
        predictions = {}
        for emotion, features in self.X_test.iteritems():
            model = self.trained_models[emotion] 
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
        self.write_results(str(self.sklearn_model), metric, eval_metrics, avg_metric)
        return eval_metrics, avg_metric

    def save(self, save_location):
        with open(save_location, "w") as f:
            pickle.dump(self.model, f)

    def write_results(self, model_name, metric, eval_metrics, avg_result):
        """ Write the results of the model to a txt file """

        with open("./results/{0}_{1}".format(model_name, metric), "w") as f:
            f.write("Evaluated {0} using {1}\n".format(model_name, metric))

            for emotion, val in eval_metrics.iteritems():
                f.write("{0} : {1}\n".format(emotion, val))

            f.write("AVG {0}\n".format(val))

