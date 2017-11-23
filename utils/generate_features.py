""" Generates features for the models """

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from scipy import sparse
from featurizer import LexiconFeaturizer
from tqdm import tqdm


def generate_lexical_features(train, test):
    """ Generate lexical features """
    print("Building affect/sentiment lexicon features for data...")

    featurizer = LexiconFeaturizer()
    train_features = {emotion: np.array([featurizer.featurize(tweet.split()) for tweet in tqdm(tweets)]) for emotion, tweets in train.iteritems()}
    test_features = {emotion: np.array([featurizer.featurize(tweet.split()) for tweet in tqdm(tweets)]) for emotion, tweets in test.iteritems()}
    return train_features, test_features


def generate_bow_features(train, test):
    """ Generate the bow features for the emotion dataset """
    print("Using BoW features")
    corpus = []
    for emotion, tweets in train.iteritems():
        corpus.extend(tweets)

    vectorizer = CountVectorizer()
    data = vectorizer.fit(corpus)
    train_data_bow = {emotion: vectorizer.transform(tweets) for emotion, tweets in train.iteritems()}
    test_data_bow = {emotion: vectorizer.transform(tweets) for emotion, tweets in test.iteritems()}
    return train_data_bow, test_data_bow


def generate_tfidf_features(train, test):
    """ Generate tfidf features for training and testing data"""
    print("Using Tf-idf features")
    corpus = []
    for emotion, tweets in train.iteritems():
        corpus.extend(tweets)

    vectorizer = TfidfVectorizer()
    data = vectorizer.fit(corpus)
    train_data_bow = {emotion: vectorizer.transform(tweets) for emotion, tweets in train.iteritems()}
    test_data_bow = {emotion: vectorizer.transform(tweets) for emotion, tweets in test.iteritems()}
    return train_data_bow, test_data_bow
 