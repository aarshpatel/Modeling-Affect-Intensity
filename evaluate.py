""" Trains and evaluates the model on the different emotions """
import argparse
import imp
import sys
import os
import utils
import numpy as np
from sklearn.model_selection import train_test_split


def load_model(model_name, X_train, y_train, X_test, y_test, optimization_parameters):
    """ Loads the model with the correct parameters """
    model_source = imp.load_source(model_name, 'models/%s.py' % (model_name))
    model = model_source.Model(X_train, y_train, X_test, y_test, optimization_parameters)
    return model


def get_labels(data):
    """ Get the labels from the data """
    labels = []
    for tweet, emotion, score in data:
        labels.append(score)
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="the model to evaluate data on")
    parser.add_argument("-f", "--features", help="the type of features to use for the model")
    parser.add_argument("-ems", "--metrics", nargs="*", help="the metrics to evaluate the model")
    parser.add_argument('-opt', "--optimize", help="whether or not to optimize the model")
    parser.add_argument("-sv" "--save", help="whether or not to save the model")

    args = parser.parse_args()

    training_data, testing_data = utils.load_data.load_training_testing_data("data/train")


    train_labels = get_labels(training_data)
    test_labels = get_labels(testing_data)


    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    print train_labels.shape
    print test_labels.shape

    train_tweets = [(" ".join(tweet), emotion) for tweet, emotion, score in training_data]
    test_tweets = [(" ".join(tweet), emotion) for tweet, emotion, score in testing_data]

    if args.features == "bow":
        X_train, X_test = utils.generate_features.generate_bow_features(X_train_corpus, X_test_corpus)
    elif args.features == "tfidf":
        X_train, X_test = utils.generate_features.generate_tfidf_features(X_train_corpus, X_test_corpus)
    else:
        X_train, X_test = None, None

    optimization_parameters = {}
    model = load_model(args.model, X_train, y_train, X_test, y_test, optimization_parameters)
    model.train()
    print "score", model.score()
    for metric in args.metrics:
        eval_score = model.evaluate(metric)
        print "{0}: {1}".format(metric, eval_score)
