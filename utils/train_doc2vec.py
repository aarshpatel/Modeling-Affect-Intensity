""" A script for training a doc2vec (paragraph vectors) model using gensim """
import pickle
import argparse 
from gensim.models import Doc2Vec 
from gensim.models.doc2vec import TaggedDocument
from preprocess_tweets import *

def load_data(data_location_path):
	tweets = []	
	with open(data_location_path, "r") as f:
		for tweet in f:
			new_tweet = preprocess_tweet(tweet)
			tweets.append(new_tweet)
	return tweets	

def preprocess_tweet(tweet):
	tweet = clean_tweet(tweet)
	return tweet

def create_tagged_documents(training_documents):
	tagged_documents = []
	for tag, tokens in enumerate(training_documents, start=1):
		td = TaggedDocument(tokens, tags=[tag])
		tagged_documents.append(td)
	return tagged_documents

def train_doc2vec_model(training_documents, doc2vec_settings, save_location):

	print "Number of docs for training: ", len(training_documents)

	print "Examle doc for training: ", training_documents[0]

	print "Starting to train doc2vec model..."
	model = Doc2Vec(training_documents, min_count=5, sample=1e-5, hs=0, dm=0, negative=5, dbow_words=1, **doc2vec_settings)
	print "Done training doc2vec model..."

	model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

	save_model(model, save_location)

def save_model(model, location):
	""" Save the doc2vec model """

	print "Saving model to: ", location	
	model.save(location + "doc2vec.bin")


if True:

	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--data", help="location of training data (.txt) file", required=True)
	parser.add_argument("-em", "--size", help="size of the embedding", required=True)
	parser.add_argument("-i", "--iter", help="number of iterations", required=True)
	parser.add_argument("-w", "--window", help="window size", required=True)
	parser.add_argument("-wrk", "--workers", help="number of workers", required=True)
	parser.add_argument("-s", "--save", help="location to save word2vec model")

	args = parser.parse_args()
	training_data_location = args.data	# location of the tweets txt file, one line per tweet eg. "../data/doc2vec/train_tweets.txt"
	model_embedding_size = int(args.size) # 300
	model_iter = int(args.iter) # 100
	model_window = int(args.window) # 3-5 (tweets are small sentences)
	model_workers = int(args.workers) # number of cores on your laptop
	save_location = args.save # string of where you want to save the doc2vec model, eg. "../data/doc2vec/doc2vec_tweets.bin"

	tweets = load_data(training_data_location)
	tagged_documents = create_tagged_documents(tweets)

	print "Number of tagged documents for training: ", len(tagged_documents)
	train_doc2vec_model(tagged_documents, {"iter": model_iter, "window": model_window, "workers": model_workers, "size": model_embedding_size}, save_location)





