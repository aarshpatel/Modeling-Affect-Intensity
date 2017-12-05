""" Trains and evaluates the model on the different emotions """
import argparse
import ConfigParser
import imp
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import utils


def load_model(model_name, X_train, y_train, optimization_parameters, sklearn_model=None):
    """
    Loads the base model model with the specific parameters

    :param model_name: the name of the model
    :param X_train: training data (# of samples x # of features)
    :param y_train: labels for the training data (# of samples * 1)
    :return: model object
   """

    model_source = imp.load_source(model_name, 'models/%s.py' % (model_name))
    model = model_source.Model(X_train, y_train, optimization_parameters, sklearn_model)
    return model


def get_labels(data):
    """
    Returns the labels for each emotion
    :param data: dictionary of training data (emotion: [emotion data])
    :return: a dictionary from emotion to a list of labels for each example for that emotion
    """

    return {emotion: np.array([val[-1] for val in values]) for emotion, values in data.iteritems()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="the model to evaluate data on")
    parser.add_argument("-f", "--features", nargs="*", help="the type of features to use for the model")
    parser.add_argument("-ems", "--metrics", nargs="*", help="the metrics to evaluate the model")
    parser.add_argument('-opt', "--optimize", help="whether or not to optimize the model")
    parser.add_argument("-sv" "--save", help="whether or not to save the model")

    args = parser.parse_args()

    if args.model == "baseline": # load the baseline parameters
        print("Loading the baseline config file...")
        config = ConfigParser.ConfigParser()
        config.read("BASELINE.ini")
        model_name = "baseline"
        sklearn_model_name = config.get("BASELINE", "model")
        features = config.get("BASELINE", "features").split(",")
        metrics = [config.get("BASELINE", "metrics")]
        optimize = False
    else:
        model_name = args.model
        features = args.features
        metrics = args.features
        optimize = True if 'optimization' in args else False
        save = True if 'save' in args else False
        sklearn_model_name = ""

    # training, testing data is a dictionary... {anger => tweets, fear => tweets, joy => tweets, sadness => tweets}
    training_data = utils.load_data.load_training_data("data/train", "data/dev")

    print("Train Data Statistics...\n")
    print("Number of anger tweets {0}".format(len(training_data["anger"])))
    print("Number of fear tweets {0}".format(len(training_data["fear"])))
    print("Number of joy tweets {0}".format(len(training_data["joy"])))
    print("Number of sadness tweets {0}".format(len(training_data["sadness"])))

    print("\n")

    y_train = get_labels(training_data)

    train_corpus = {emotion: [" ".join(val[0]) for val in values]for emotion, values in training_data.iteritems()}

    print("Featurizers being used: ")
    for idx, feature in enumerate(features, start=1):
        print("   {0}. {1}".format(idx, feature))

    featurizer = utils.generate_features.Featurizer(features, train_corpus)
    X_train = featurizer.generate_all_features()

    print("Feature length: {0}".format(X_train["anger"].shape[1]))
    print("\n")

    optimization_parameters = {
        'RandomForestRegressor': {
            'n_estimators': [10, 20, 30, 50, 100, 500]
        },
        'SVR': {'C': [0.001, 0.01, 0.1, 1, 10]},
        '': ''
    }

    if sklearn_model_name:
        print("Using model: {0}".format(sklearn_model_name))
    else:
        print("Using model: {0}".format(model_name))

    print("\n")

    # load the model (you don't need to pass in a sklearn model)
    if sklearn_model_name:
        # load a model that uses a sklearn model
        model = load_model(model_name, X_train, y_train, optimization_parameters[sklearn_model_name], eval(sklearn_model_name))
    else:
        # load a model that uses keras
        model = load_model(model_name, X_train, y_train, optimization_parameters[sklearn_model_name], None)

    if optimize == 'True':
        best_model_params = model.optimize("pearson_correlation") # which metric to optimize models
        emotions_cv_scores = model.train(best_model_params)
    else:
        emotions_cv_scores = model.train()

    print("10-Fold CV Scores (Pearson Correlation) ")
    for emotion, score in emotions_cv_scores.iteritems():
        print("Emotion: {0}, Score: {1}".format(emotion, score))

    print("AVG Pearson Correlation: ", np.mean(emotions_cv_scores.values()))
