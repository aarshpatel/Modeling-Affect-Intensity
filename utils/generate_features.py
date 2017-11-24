""" Generates features for the models """

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from scipy import sparse
from lexicon_featurizer import LexiconFeaturizer
from embedding_featurizer import EmbeddingFeaturizer
from tqdm import tqdm
from scipy import sparse

class Featurizer(object):

    def __init__(self, featurizers, train, test):
        """ NOTE: you should only combine dense vectors, so don't pass in bow, and tfidf together """
        self.featurizers = featurizers
        self.train = train
        self.test = test
        self.featurizers_implemented = {
            "lexicons": self.generate_lexical_features,
            "glove": self.generate_glove_embedding_features,
            "bow": self.generate_bow_features,
            "tfidf": self.generate_tfidf_features 
        }

    def generate_lexical_features(self):
        """ Generate lexical features """
        print("Building affect/sentiment lexicon features for data...")
        featurizer = LexiconFeaturizer()
        train_features = {emotion: np.array([featurizer.featurize(tweet.split()) for tweet in tqdm(tweets)]) for emotion, tweets in self.train.iteritems()}
        test_features = {emotion: np.array([featurizer.featurize(tweet.split()) for tweet in tqdm(tweets)]) for emotion, tweets in self.test.iteritems()}
        return train_features, test_features

    def generate_glove_embedding_features(self):
        """ Generate glove embedding features """
        print("Building Glove Embedding features for data...")
        embedding_featurizer = EmbeddingFeaturizer(glove=True)
        train_features = {emotion: np.array([embedding_featurizer.glove_embeddings_for_tweet(tweet.split()) for tweet in tqdm(tweets)]) for emotion, tweets in self.train.iteritems()}
        test_features = {emotion: np.array([embedding_featurizer.glove_embeddings_for_tweet(tweet.split()) for tweet in tqdm(tweets)]) for emotion, tweets in self.test.iteritems()}
        return train_features, test_features
        
    def generate_bow_features(self):
        """ Generate the bow features for the emotion dataset """
        print("Using BoW features")
        corpus = []
        for emotion, tweets in self.train.iteritems():
            corpus.extend(tweets)

        vectorizer = CountVectorizer()
        vectorizer.fit(corpus)
        train_data_bow = {emotion: vectorizer.transform(tweets) for emotion, tweets in self.train.iteritems()}
        test_data_bow = {emotion: vectorizer.transform(tweets) for emotion, tweets in self.test.iteritems()}
        return train_data_bow, test_data_bow

    def generate_tfidf_features(self):
        """ Generate tfidf features for training and testing data"""
        print("Using Tf-idf features")
        corpus = []
        for emotion, tweets in self.train.iteritems():
            corpus.extend(tweets)

        vectorizer = TfidfVectorizer()
        vectorizer.fit(corpus)
        train_data_bow = {emotion: vectorizer.transform(tweets) for emotion, tweets in self.train.iteritems()}
        test_data_bow = {emotion: vectorizer.transform(tweets) for emotion, tweets in self.test.iteritems()}
        return train_data_bow, test_data_bow

    def generate_all_features(self):
        """ 
        Concatenate all featurizes (only for dense features)
        Eg. featurizers=["lexicons", "glove"] ==> lexicon_features + glove features
        Do not do this for bow + tfidf (both are sparse matrices)
        """

        train_full =  {}
        test_full = {}

        for featurizer in self.featurizers:
            print("featurizer:", featurizer)
            train_features, test_features = self.featurizers_implemented[featurizer].__call__()

            for emotion, values in train_features.iteritems():
                print emotion, values.shape

            for emotion, train_feats in train_features.iteritems():
                if emotion not in train_full:
                    train_full[emotion] = train_feats
                else:
                   # concatentate the matrices
                   concat_features = np.hstack((train_full[emotion], train_feats)) 
                   train_full[emotion] = concat_features

            for emotion, test_feats in test_features.iteritems():
                if emotion not in test_full:
                    test_full[emotion] = test_feats 
                else:
                   # concatentate the matrices
                   concat_features = np.hstack((test_full[emotion], test_feats)) 
                   test_full[emotion] = concat_features

        return train_full, test_full






 