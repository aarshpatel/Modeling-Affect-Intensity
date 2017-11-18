""" Trains and evaluates the model on the different emotions """
import argparse
import imp
import sys
import os
# sys.path.append('/Users/aarsh/Documents/Modeling-Affect-Intensity/utils')
import utils
import numpy as np

def load_model(model_name, X_train, y_train, X_test, y_test, optimization_parameters):
	""" Loads the model with the correct parameters """
	model_source = imp.load_source(model_name, 'models/%s.py' % (model_name))
	model = model_source.Model(X_train, y_train, X_test, y_test, optimization_parameters={})
	return model

def evaluate_model(model, evaluation_metric):
	""" Given the model name, optimizations, feature_types, data ==> train the model """
	evaluation_score = model.evaluate(evaluation_metric)
	return evaluation_score

def get_labels(data):
	""" Get the labels from the data """
	labels = []
	for tweet, emotion, score in data:
		labels.append(score)
	return np.asarray(labels)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-tr", "--train", help="location of training data")
	parser.add_argument("-te", "--test", help="location of test data")
	parser.add_argument("-m", "--model", help="the model to evaluate data on")
	parser.add_argument("-f", "--features", help="the type of features to use for the model")
	parser.add_argument("-me", "--metric", help="the metric to evaluate models")
	parser.add_argument('-opt', "--optimize", help="whether or not to optimize the models")
	parser.add_argument("-sv" "--save", help="save the model")

	args = parser.parse_args()

	print "Train Data:", args.train
	print "Test Data: ", args.test
	print "Model: ", args.model
	print "Features: ", args.features
	print "Metric: ", args.metric
	print "\n"

	training_data = utils.load_data.load_training_data("./data/train")	

	X_train_samples = [" ".join(tweet) for tweet, emotion, score in training_data]
	X_test_samples = [" ".join(tweet) for tweet, emotion, score in training_data]	

	y_train = get_labels(training_data)
	y_test = get_labels(training_data) # change to training data 

	if args.features == "bow":
		X_train, X_test =  utils.generate_features.generate_bow_features(X_train_samples, X_test_samples)
	elif args.features == "tfidf":
		X_train, X_test = utils.generate_features.generate_tfidf_features(X_train_samples, X_test_samples)
	else:
		X_train, X_test = None, None

	optimization_parameters = {}	
	model = load_model(args.model, X_train, y_train, X_test, y_test, optimization_parameters)
	model.train()
	eval_score = evaluate_model(model, args.metric)

	print "{0}: {1}".format(args.metric, eval_score)

