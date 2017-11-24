import re
import string
from nltk.corpus import stopwords
from tweetokenize import Tokenizer

stopWords = set(stopwords.words('english'))
# tweet_reg = '([@][A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)'

def clean_tweet(tweet):
	""" Simple tweet preprocessing """
	gettokens = Tokenizer()
	tweet = " ".join(gettokens.tokenize(tweet))
	# tweet = re.sub(tweet_reg, "", tweet)
	# tweet = re.sub("\d+", "", tweet)
	# tweet = tweet.lower().strip()
	tweet = [word for word in tweet.split() if word not in stopWords]
	tweet = [word for word in tweet if word not in string.punctuation]
	return tweet

