""" Trains and evaluates the model on the different emotions """
import argparse
import imp
import sys
import os
import utils
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_model(model_name, X_train, y_train, X_test, y_test, optimization_parameters):
    """ Loads the model with the correct parameters """
    model_source = imp.load_source(model_name, 'models/%s.py' % (model_name))
    model = model_source.Model(X_train, y_train, X_test, y_test, optimization_parameters)
    return model

def get_labels(data):
    """ Returns the labels for each emotion """
    return {emotion: np.array([val[-1] for val in values]) for emotion, values in data.iteritems()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="the model to evaluate data on")
    parser.add_argument("-f", "--features", help="the type of features to use for the model")
    parser.add_argument("-ems", "--metrics", nargs="*", help="the metrics to evaluate the model")
    parser.add_argument('-opt', "--optimize", help="whether or not to optimize the model")
    parser.add_argument("-sv" "--save", help="whether or not to save the model")

    args = parser.parse_args()

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


    if args.features == "bow":
        X_train, X_test = utils.generate_features.generate_bow_features(train_corpus, test_corpus)

    elif args.features == "tfidf":
        X_train, X_test = utils.generate_features.generate_tfidf_features(train_corpus, test_corpus)

    elif args.features == "lexicons":
        X_train, X_test = utils.generate_features.generate_lexical_features(train_corpus, test_corpus) 
    else:
        X_train, X_test = None, None

    print(X_train["anger"].shape)
    print(X_test["anger"].shape)

    optimization_parameters = {}
    model = load_model(args.model, X_train, y_train, X_test, y_test, optimization_parameters)
    model.train()
    for metric in args.metrics:
        predictions = model.predict()
        eval_score, avg_eval = model.evaluate(predictions, y_test, metric)
        for emotion, score in eval_score.iteritems():
            print emotion, score 
        print("AVG {0}: {1}".format(metric, avg_eval))
