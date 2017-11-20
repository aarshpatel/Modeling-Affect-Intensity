""" Generates features for the models """

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from scipy import sparse


def generate_bow_features(X_train_samples, X_test_samples, train_emotions, test_emotions):
    """ Generate the bow features for the X_train, X_test """

    print("Using BoW features")
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train_samples)
    X_train_dense = X_train.todense()
    X_train_new = np.empty((X_train_dense.shape[1]+1))
    for idx, train_pt in enumerate(X_train_dense):
        train_pt_list = train_pt.tolist()[0]
        train_pt_list.append(train_emotions[idx])
        np.append(X_train_new, np.asarray(train_pt_list), axis=0)
    X_train = sparse.csr_matrix(X_train_dense)

    X_test = vectorizer.transform(X_test_samples)
    X_test_dense = X_test.todense()
    X_test_new = np.empty((X_test_dense.shape[1] + 1))
    for idx, test_pt in enumerate(X_test_dense):
        test_pt_list = test_pt.tolist()
        test_pt_list.append(test_emotions[idx])
        np.append(X_test_new, np.asarray(test_pt_list), axis=0)
    X_test = sparse.csr_matrix(X_test_dense)

    return (X_train, X_test)


def generate_tfidf_features(X_train_samples, X_test_samples, train_emotions, test_emotions):
    """ Generate tfidf features """
    print("Using Tf-idf features")
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train_samples)
    X_train_dense = X_train.todense()
    X_train_new = np.empty((X_train_dense.shape[1] + 1))
    for idx, train_pt in enumerate(X_train_dense):
        train_pt_list = train_pt.tolist()[0]
        train_pt_list.append(train_emotions[idx])
        np.append(X_train_new, np.asarray(train_pt_list), axis=0)
    X_train = sparse.csr_matrix(X_train_dense)

    X_test = vectorizer.transform(X_test_samples)
    X_test_dense = X_test.todense()
    X_test_new = np.empty((X_test_dense.shape[1] + 1))
    for idx, test_pt in enumerate(X_test_dense):
        test_pt_list = test_pt.tolist()
        test_pt_list.append(test_emotions[idx])
        np.append(X_test_new, np.asarray(test_pt_list), axis=0)
    X_test = sparse.csr_matrix(X_test_dense)

    return (X_train, X_test)
