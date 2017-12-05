import utils
import pickle
from models import base_model
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold

class Model(base_model.Model):
    """
    Implementation of CNN model that uses glove embeddings as input 
    Take the avg embedding of the tweets and feed that into a feed forward neural network to compute scores
    """

    def __init__(self, X_train, y_train, optimization_parameters={}, sklearn_model=None):
        base_model.Model.__init__(self, X_train, y_train, optimization_parameters)

    def train(self, metric, optimization_parameters=None):
        """ Train a model that predict affect intenity based on avg embedding of the tweets """

        if metric == "rmse":
            scoring_metric = utils.metrics.rmse_scorer
        elif metric == "mse":
            scoring_metric = utils.metrics.mse_scorer
        elif metric == "pearson_correlation":
            scoring_metric = utils.metrics.pearson_scorer

        input_dim = self.X_train["anger"].shape[1] # just get the dim of the features

        self.trained_models = {}
        self.emotion_to_cv_score = {}

        for emotion, features in self.X_train.iteritems():
            n_folds = 5

            model = self._build_keras_model(input_dim)
            print("Model: ", model)
            clf = KerasRegressor(build_fn=model, nb_epoch=30, batch_size=8, verbose=1, validation_split=0.2)
            # kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            # results = cross_val_score(clf, features, self.y_train[emotion], cv=kfold, scoring=scoring_metric)
            cv_score_avg = results.mean()
            self.emotion_to_cv_score[emotion] = cv_score_avg

        return self.emotion_to_cv_score

    def _build_keras_model(self, input_dim):
        """ Build the keras model """
        model = Sequential()
        model.add(Dense(300, input_dim=input_dim, kernel_initializer='normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(150, kernel_initializer='normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(50, kernel_initializer='normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(25, kernel_initializer='normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(1)) # sigmoid layer (return affect intensity score)
        model.compile(loss='mean_squared_error', optimizer = 'adam') # use the pearson correlation as the loss
        return model

    def optimize(self, metric):
        """
        Optimize the baseline model. We don't want to optimize our baseline model
        @metric: the metric to optimize model

        TODO: how to optimize sklearn regression DNN model
        """
        pass

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

