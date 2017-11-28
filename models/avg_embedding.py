import utils
import pickle
from models import base_model
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasRegressor

class Model(base_model.Model):
    """ 
    Implementation of AVG Embedding (uses glove embeddings)
    Take the avg embedding of the tweets and feed that into a feed forward neural network to compute scores
    """

    def __init__(self, X_train, y_train, X_test, y_test, optimization_parameters={}, sklearn_model=None):
        base_model.Model.__init__(self, X_train, y_train, X_test, y_test, optimization_parameters)

    def train(self, optimization_parameters=None):
        """ Train a model that predict affect intenity based on avg embedding of the tweets """
        input_dim = X_train["angry"].shape
        self.trained_models = {}
        for emotion, features in self.X_train.iteritems():
            model = self._build_keras_model(input_dim) 
            clf = KerasRegressor(build_fn=model, nb_epoch=50, batch_size=16,verbose=1)
            clf.fit(features, self.y_train[emotion])
            self.trained_models[emotion] = clf

    def _build_keras_model(self, input_dim):
        """ Build the keras model """
        model = Sequential()
        model.add(Dense(50, input_dim=input_dim, init='normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # model.add(Dense(20))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer = 'adam')
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

