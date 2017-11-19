import os
from preprocess_tweets import *
from sklearn.model_selection import train_test_split
from random import shuffle

def load_tweets_for_emotion(loc, emotion):
	""" Load tweets and scores for a specific emotion """
	tweets = []
	with open(loc, "r") as f:
		for line in f.readlines():
			temp = line.split()
			score = float(temp[-1])
			affect = temp[-2]
			tweet = clean_tweet(" ".join(temp[1:-2]))
			tweets.append((tweet, emotion, score))
	return tweets

def load_training_testing_data(loc):
	""" 
	Load all of the training and testing data 
	Returns a 80-20 training/testing split
	"""

	training_data = []
	testing_data = []
	
	for filename in os.listdir(loc):
		if filename.endswith(".txt"):
			emotion = filename.rstrip(".txt")
			file_location = loc + "/" + filename
			tweets_for_emotion = load_tweets_for_emotion(file_location, emotion)
			# shuffle the tweets for the emotion
			shuffle(tweets_for_emotion)

			training_data.extend(tweets_for_emotion)
	return training_data


