""" Trains and evaluates the model on the different emotions """
import argparse
import imp
import sys
import os
# sys.path.append('/Users/aarsh/Documents/Modeling-Affect-Intensity/utils')
import utils
import numpy as np
from sklearn.model_selection import train_test_split


def load_model(model_name, X_train, y_train, X_test, y_test, optimization_parameters):
    """ Loads the model with the correct parameters """
    model_source = imp.load_source(model_name, 'models/%s.py' % (model_name))
    model = model_source.Model(X_train, y_train, X_test, y_test, optimization_parameters)
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

    train_input = utils.load_data.load_training_data("data/train")
    dev_input = utils.load_data.load_training_data("data/dev")

    train_emotions, train_labels = get_emotions_and_labels(train_input)
    dev_emotions, dev_labels = get_emotions_and_labels(dev_input)

    # train_input.extend(dev_input)
    # train_labels.extend(dev_labels)
    # train_emotions.extend(dev_emotions)

    # X_train, X_test, y_train, y_test = train_test_split(train_input, train_labels, test_size=0.2, random_state=10)

    # y_train = np.asarray(y_train)
    # y_test = np.asarray(y_test)

    emotion_to_feature = {
        "anger": 0,
        "fear": 1,
        "joy": 2,
        "sadness": 3
    }

    train_emotions = [emotion_to_feature[emotion] for emotion in train_emotions]
    train_emotions = np.asarray(train_emotions)

    dev_emotions = [emotion_to_feature[emotion] for emotion in dev_emotions]
    dev_emotions = np.asarray(dev_emotions)

    y_train = train_labels
    y_test = dev_labels

    X_train_corpus = [" ".join(tweet) for tweet, emotion, score in train_input]
    X_dev_corpus = [" ".join(tweet) for tweet, emotion, score in dev_input]

    if args.features == "bow":
        X_train, X_test = utils.generate_features.generate_bow_features(X_train_corpus, X_dev_corpus, train_emotions, dev_emotions)
    elif args.features == "tfidf":
        X_train, X_test = utils.generate_features.generate_tfidf_features(X_train_corpus, X_dev_corpus, train_emotions, dev_emotions)
    else:
        X_train, X_test = None, None

    print type(X_train)
    print type(X_test)

    optimization_parameters = {}
    model = load_model(args.model, X_train, y_train, X_test, y_test, optimization_parameters)
    model.train()
    print "score", model.score()
    for metric in args.metrics:
        eval_score = model.evaluate(metric)
        print "{0}: {1}".format(metric, eval_score)
