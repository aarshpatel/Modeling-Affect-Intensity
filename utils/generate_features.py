""" Generates features for the models """

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from scipy import sparse


def generate_bow_features(X_train_samples, X_test_samples, train_emotions, test_emotions):
    """ Generate the bow features for the X_train, X_test """
    print("Using BoW features")
    vectorizer = CountVectorizer()
    vectorizer, X_train = add_emotion_feature(vectorizer, X_train_samples, train_emotions, train=True)
    vectorizer, X_test = add_emotion_feature(vectorizer, X_test_samples, test_emotions, train=False)
    return X_train, X_test


def generate_tfidf_features(X_train_samples, X_test_samples, train_emotions, test_emotions):
    """ Generate tfidf features for training and testing data"""
    print("Using Tf-idf features")
    vectorizer = TfidfVectorizer()
    vectorizer, X_train = add_emotion_feature(vectorizer, X_train_samples, train_emotions, train=True)
    vectorizer, X_test = add_emotion_feature(vectorizer, X_test_samples, test_emotions, train=False)
    return X_train, X_test

def add_emotion_feature(vectorizer, data, emotions, train=False):
    """
    Add a emotion feature to each feature vector
    Only works with Bow and Tfidf Vectorizers
    """

    if train:
        # if train then do a fit transform on the data
        data = vectorizer.fit_transform(data)
    else:
        # if train is false, then just do a transform
        data = vectorizer.transform(data)

    data_dense = data.todense()
    data_new = [] 
    for idx, test_pt in enumerate(data_dense):
        test_pt_list = test_pt.tolist()[0]
        test_pt_list.append(emotions[idx])
        data_new.append(test_pt_list)
    data_new = np.array(data_new)
    data = sparse.csr_matrix(data_new)
    return vectorizer, data

def tweet_embedding(tweet):
    """ Get the embedding representing the tweet """
    return tweet
     
