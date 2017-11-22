""" Trains and evaluates the model on the different emotions """
import argparse
import imp
import sys
import os
import utils
import numpy as np
from sklearn.model_selection import train_test_split


def load_model(model_name, X_train, y_train, X_test, y_test, train_emotions, test_emotions, optimization_parameters):
    """ Loads the model with the correct parameters """
    model_source = imp.load_source(model_name, 'models/%s.py' % (model_name))
    model = model_source.Model(X_train, y_train, X_test, y_test, train_emotions, test_emotions, optimization_parameters)
    return model


def get_emotions_and_labels(data):
    """ Get the labels from the data """
    emotions = []
    labels = []
    for tweet, emotion, score in data:
        emotions.append(emotion)
        labels.append(score)
    return emotions, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="the model to evaluate data on")
    parser.add_argument("-f", "--features", help="the type of features to use for the model")
    parser.add_argument("-ems", "--metrics", nargs="*", help="the metrics to evaluate the model")
    parser.add_argument('-opt', "--optimize", help="whether or not to optimize the model")
    parser.add_argument("-sv" "--save", help="whether or not to save the model")

    args = parser.parse_args()

    training_data, testing_data = utils.load_data.load_training_testing_data("data/train")

    train_emotions, train_labels = get_emotions_and_labels(training_data)
    test_emotions, test_labels = get_emotions_and_labels(testing_data)

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    emotion_to_feature = {
        "anger": 0,
        "fear": 1,
        "joy": 2,
        "sadness": 3
    }

    train_emotions = [emotion_to_feature[emotion] for emotion in train_emotions]
    train_emotions = np.asarray(train_emotions)

    test_emotions = [emotion_to_feature[emotion] for emotion in test_emotions]
    test_emotions = np.asarray(test_emotions)

    X_train_corpus = [" ".join(tweet) for tweet, emotion, score in training_data]
    X_test_corpus = [" ".join(tweet) for tweet, emotion, score in testing_data]

    print "X_train corpus", len(X_train_corpus)
    print "X_test corpus", len(X_test_corpus)

    if args.features == "bow":
        X_train, X_test = utils.generate_features.generate_bow_features(X_train_corpus, X_test_corpus, train_emotions, test_emotions)
    elif args.features == "tfidf":
        X_train, X_test = utils.generate_features.generate_tfidf_features(X_train_corpus, X_test_corpus, train_emotions, test_emotions)
    else:
        X_train, X_test = None, None

    y_train = train_labels
    y_test = test_labels

    print X_train.shape
    print y_train.shape
    print("-------------------")
    print X_test.shape
    print y_test.shape

    optimization_parameters = {}
    model = load_model(args.model, X_train, y_train, X_test, y_test, train_emotions, test_emotions, optimization_parameters)
    model.train()
    print "score", model.score()
    for metric in args.metrics:
        predictions = model.predict()
        eval_score = model.evaluate(predictions, y_test, metric)
        print "{0}: {1}".format(metric, eval_score)

    model.get_emotion_test_results("rmse")
