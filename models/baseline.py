""" Implementation of baseline model using any set of features """
import utils
import pickle
from models import base_model
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import sklearn.model_selection as ms


class Model(base_model.Model):
    """ Implementation of baseline model """

    def __init__(self, X_train, y_train, optimization_parameters={}, sklearn_model=None):
        base_model.Model.__init__(self, X_train, y_train, optimization_parameters)
        self.sklearn_model = sklearn_model

    def train(self, metric, optimization_parameters=None):
        """ Train the baseline model on all emotions """

        self.trained_models = {}
        self.emotion_to_cv_score = {}

        if metric == "rmse":
            scoring_metric = utils.metrics.rmse_scorer
        elif metric == "mse":
            scoring_metric = utils.metrics.mse_scorer
        elif metric == "pearson_correlation":
            scoring_metric = utils.metrics.pearson_scorer

        for emotion, features in self.X_train.iteritems():
            model = self.sklearn_model() # use the default hyperparameters
            cv_score = cross_val_score(model, features, self.y_train[emotion], scoring=scoring_metric, cv=ms.StratifiedKFold(shuffle=True, n_splits=10), n_jobs=-1)
            cv_score_avg = np.mean(cv_score)
            self.emotion_to_cv_score[emotion] = cv_score_avg
            self.trained_models[emotion] = model.fit(features, self.y_train[emotion])
            # if optimization_parameters:
            #     model.set_params(**optimization_parameters[emotion])
        return self.emotion_to_cv_score

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

    def predict(self, X_test):
        """ Predict on new test data """
        predictions = {}
        for emotion, features in X_test.iteritems():
            model = self.trained_models[emotion]
            emotion_predictions = model.predict(features)
            predictions[emotion] = emotion_predictions
        return predictions

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
