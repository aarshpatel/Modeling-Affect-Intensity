""" Generates features for the models """

from sklearn.feature_extraction.text import CountVectorizer

def generate_bow_features(X_train_samples, X_test_samples):
	""" Generate the bow features for the X_train, X_test """
	vectorizer = CountVectorizer()
	X_train = vectorizer.fit_transform(X_train_samples)
	X_test = vectorizer.transform(X_test_samples)
	return (X_train, X_test)


