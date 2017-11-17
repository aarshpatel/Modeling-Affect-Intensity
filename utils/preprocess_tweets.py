import preprocessor as p
import re
import string
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))
tweet_reg = '([@][A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)'

def clean_tweet(tweet):
	""" Simple tweet preprocessing """
	tweet = re.sub(tweet_reg, "", tweet)
	tweet = re.sub("\d+", "", tweet)
	tweet = tweet.lower().strip()
	tweet = [word for word in tweet.split() if word not in stopWords]
	return tweet

