import os
from preprocess_tweets import *

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

def load_training_data(loc):
	""" Load all of the training data """
	training_data = []
	for filename in os.listdir(loc):
		if filename.endswith(".txt"):
			emotion = filename.rstrip(".txt")
			file_location = loc + "/" + filename
			tweets_for_emotion = load_tweets_for_emotion(file_location, emotion)
			training_data.extend(tweets_for_emotion)
	return training_data


