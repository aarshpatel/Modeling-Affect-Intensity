import os
from preprocess_tweets import *
from random import shuffle


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

            train = tweets_for_emotion[0:int(len(tweets_for_emotion) * .80)]
            test = tweets_for_emotion[int(len(tweets_for_emotion) * .80):]

            training_data.extend(train)
            testing_data.extend(test)

    return training_data, testing_data


def load_tweets_for_emotion(loc, emotion):
    """ Load tweets and scores for a specific emotion """
    tweets = []
    with open(loc, "r") as f:
        for line in f.readlines():
            temp = line.split()
            score = float(temp[-1])
            tweet = clean_tweet(" ".join(temp[1:-2]))
            tweets.append((tweet, emotion, score))
    return tweets
