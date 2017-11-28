""" Trains and evaluates the model on the different emotions """
import argparse
import imp
import utils
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


def load_model(model_name, X_train, y_train, X_test, y_test, optimization_parameters, sklearn_model):
    """ Loads the model with the correct parameters """
    model_source = imp.load_source(model_name, 'models/%s.py' % (model_name))
    model = model_source.Model(X_train, y_train, X_test, y_test, optimization_parameters, sklearn_model)
    return model


def get_labels(data):
    """ Returns the labels for each emotion """
    return {emotion: np.array([val[-1] for val in values]) for emotion, values in data.iteritems()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="the model to evaluate data on")
    parser.add_argument("-f", "--features", nargs="*", help="the type of features to use for the model")
    parser.add_argument("-ems", "--metrics", nargs="*", help="the metrics to evaluate the model")
    parser.add_argument('-opt', "--optimize", help="whether or not to optimize the model")
    parser.add_argument("-sv" "--save", help="whether or not to save the model")

    args = parser.parse_args()
    print(args)
    # training, testing data is a dictionary... {anger => tweets, fear => tweets, joy => tweets, sadness => tweets}
    training_data, testing_data = utils.load_data.load_training_testing_data("data/train", "data/dev")

    print("Train Data Statistics...\n")
    print("Anger: ", len(training_data["anger"]))
    print("Fear:", len(training_data["fear"]))
    print("Joy: ", len(training_data["joy"]))
    print("Sadness: ", len(training_data["sadness"]))

    print('\n')

    print("Test Data Statistics...\n")
    print("Anger: ", len(testing_data["anger"]))
    print("Fear: ", len(testing_data["fear"]))
    print("Joy: ", len(testing_data["joy"]))
    print("Sadness: ", len(testing_data["sadness"]))
    print('\n')

    y_train = get_labels(training_data)
    y_test = get_labels(testing_data)

    train_corpus = {emotion: [" ".join(val[0]) for val in values]for emotion, values in training_data.iteritems()}
    test_corpus = {emotion: [" ".join(val[0]) for val in values]for emotion, values in testing_data.iteritems()}

    featurizer = utils.generate_features.Featurizer(args.features, train_corpus, test_corpus)
    X_train, X_test = featurizer.generate_all_features()

    optimization_parameters = {
        'RandomForestRegressor': {
            'n_estimators': [10, 20, 30, 50, 100, 500]
        },
        'SVR': {'C': [0.001, 0.01, 0.1, 1, 10]}
    }

    # load the model (you don't need to pass in a sklearn model)
    model = load_model(args.model, X_train, y_train, X_test, y_test, optimization_parameters["RandomForestRegressor"], RandomForestRegressor)

    if args.optimize == 'True':
        best_model_params = model.optimize("pearson_correlation") # which metric to optimize models
        print(best_model_params)
        model.train(best_model_params)
    else:
        model.train()

    for metric in args.metrics:
        predictions = model.predict()
        eval_score, avg_eval = model.evaluate(predictions, y_test, metric)
        print("\n")
        for emotion, score in eval_score.iteritems():
            print emotion, score
        print("AVG {0}: {1}".format(metric, avg_eval))
