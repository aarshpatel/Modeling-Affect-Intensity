""" Generates features for the models """

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

emotion_to_feature = {
	"anger": 0,
	"fear": 1,
	"joy": 2,
	"sadness": 3
}

def generate_bow_features(X_train_samples, X_test_samples):
    """ Generate the bow features for the X_train, X_test """
    print("Using BoW features")
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train_samples)
    X_train_dense = X_train.todense()
    X_train_new = np.empty((0, X_train_dense.shape[1]))
    for idx, train_pt in enumerate(X_train_dense):
        train_pt_list = train_pt.tolist()
        train_pt_list.append(train_emotions[idx])
        print "shape", X_train_dense[idx].shape
        np.append(X_train_new, np.asarray(train_pt_list), axis=0)
    X_train = sparse.csr_matrix(X_train_dense)
    X_test = vectorizer.transform(X_test_samples)
    return (X_train, X_test)


def generate_tfidf_features(X_train_samples, X_test_samples):
    """ Generate tfidf features """
    print("Using Tf-idf features")
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train_samples)
    X_test = vectorizer.transform(X_test_samples)
    return (X_train, X_test)
