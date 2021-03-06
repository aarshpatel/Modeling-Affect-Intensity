import os
from preprocess_tweets import *
from random import shuffle


def load_data_from_dir(data_loc):
    data = {}
    for filename in os.listdir(data_loc):
        if filename.endswith(".txt"):
            emotion = filename.rstrip(".txt")
            file_location = data_loc + "/" + filename
            tweets_for_emotion = load_tweets_for_emotion(file_location, emotion)
            data[emotion] = tweets_for_emotion
    return data


def load_training_data(training_data_loc, dev_data_loc):
    """
    Load all of the training and testing data (which is the dev data in our case)
    """

    training_data = load_data_from_dir(training_data_loc)
    dev_data = load_data_from_dir(dev_data_loc)
    # combine the training and dev data into one
    for emotion, tweet_data in dev_data.iteritems():
        training_data[emotion].extend(tweet_data)
    return training_data


def load_tweets_for_emotion(loc, emotion):
    """ Load tweets and scores for a specific emotion """
    tweets = []
    with open(loc, "r") as f:
        for line in f.readlines():
            temp = line.split()
            score = float(temp[-1])
            tweet = clean_tweet(" ".join(temp[1:-2]))
            tweets.append((tweet, score))
    return tweets
