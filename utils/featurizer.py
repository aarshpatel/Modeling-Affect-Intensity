from collections import defaultdict
import gzip

class LexiconFeaturizer(object):
    """ A class that featurizes a tweet affect/sentiment lexicons"""

    def __init__(self, list_of_featurizers=[]):
        self.list_of_featurizers = []

    def get_bigrams(self, tokens):
        """ Return a list of bigram from a set of tokens """
        return [a + " " + b for a, b in zip(tokens, tokens[1:])]

    def nrc_hashtag_emotion(self, tokens):
        """ Build features using NRC Hashtag emotion dataset """
        nrc_hashtag_emotion_path = "../data/lexicons/NRC-Hashtag-Emotion-Lexicon-v0.2.txt.gz"
        lexicon_map = defaultdict(list)

        with gzip.open(nrc_hashtag_emotion_path, 'rb') as f:
            lines = f.read().splitlines()
            for l in lines[1:]:
                splits = l.decode('utf-8').split('\t')
                lexicon_map[splits[0]] = [float(num) for num in splits[1:]]
        num_features = 10  # 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust'
        sum_vec = [0.0] * num_features
        for token in tokens:
            if token in lexicon_map:
                sum_vec = [a + b for a, b in zip(sum_vec, lexicon_map[token])] # sum up the individual word feature vectors
        feature_names = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust']
        feature_names = ['nrc_hashtag_emotion_' + name for name in feature_names]
        return dict(zip(feature_names, sum_vec))

    def nrc_affect_intensity(self, tokens):
        """ Build feature vector using NRC affect intensity lexicons """
        nrc_affect_intensity_path = "../data/lexicons/nrc_affect_intensity.txt.gz"
        lexicon_map = defaultdict(list)

        with gzip.open(nrc_affect_intensity_path, 'rb') as f:
            lines = f.read().splitlines()
            for l in lines[1:]:
                splits = l.decode('utf-8').split('\t')
                lexicon_map[splits[0]] = [float(num) for num in splits[1:]]

        num_features = 10  # 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust'
        sum_vec = [0.0] * num_features
        for token in tokens:
            if token in lexicon_map:
                sum_vec = [a + b for a, b in zip(sum_vec, lexicon_map[token])]
        
        feature_names = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust']
        feature_names = ['nrc_affect_intensity_' + name for name in feature_names]
        return dict(zip(feature_names, sum_vec))

    def nrc_hashtag_sentiment_lexicon_unigrams(self, tokens):
        """
        Function returns sum of intensities of
        positive and negative tokens using only unigrams. Also returns
        the number of positive and negative tokens
        """

        nrc_hashtag_sentiment_lexicon_unigrams_path = "../data/lexicons/NRC-Hashtag-Sentiment-Lexicon-v0.1/unigrams-pmilexicon.txt.gz"
        with gzip.open(nrc_hashtag_sentiment_lexicon_unigrams_path, 'rb') as f:
            lines = f.read().splitlines()
            lexicon_map = {}
            for l in lines:
                splits = l.decode('utf-8').split('\t')
                lexicon_map[splits[0]] = float(splits[1])

        positive_score, negative_score = 0.0, 0.0
        positive_unigram_words, negative_unigram_words = 0, 0

        for token in tokens:
            if token in lexicon_map:
                if lexicon_map[token] >= 0:
                    positive_score += lexicon_map[token]
                    positive_unigram_words += 1
                else:
                    negative_score += lexicon_map[token]
                    negative_unigram_words += 1

        return {
            "nrc_hashtag_sentiment_positive_unigram_score": positive_score, 
            "nrc_hastag_sentiment_negative_unigram_score": negative_score, 
            "nrc_hashtag_sentiment_postive_unigram_words": positive_unigram_words, 
            "nrc_hashtag_sentiment_negative_unigram_words": negative_unigram_words
        }

    def nrc_hashtag_sentiment_lexicon_bigrams(self, tokens):
        """ 
        Function returns sum of intensities of 
        positive and negative tokens using only unigrams. Also returns
        the number of positive and negative tokens
        """

        nrc_hashtag_sentiment_lexicon_bigrams_path = "../data/lexicons/NRC-Hashtag-Sentiment-Lexicon-v0.1/bigrams-pmilexicon.txt.gz"
        with gzip.open(nrc_hashtag_sentiment_lexicon_bigrams_path, 'rb') as f:
            lines = f.read().splitlines()
            lexicon_map = {}
            for l in lines:
                splits = l.decode('utf-8').split('\t')
                lexicon_map[splits[0]] = float(splits[1])

        positive_score, negative_score = 0.0, 0.0
        positive_bigram_words, negative_bigram_words = 0, 0

        # loop through the bigrams  
        for token in self.get_bigrams(tokens):
            if token in lexicon_map:
                if lexicon_map[token] >= 0:
                    positive_score += lexicon_map[token]
                    positive_bigram_words += 1
                else:
                    negative_score += lexicon_map[token]
                    negative_bigram_words += 1
        return {
            "nrc_hashtag_sentiment_positive_bigram_score": positive_score, 
            "nrc_hastag_sentiment_negative_bigram_score": negative_score, 
            "nrc_hashtag_sentiment_postive_bigram_words": positive_bigram_words, 
            "nrc_hashtag_sentiment_negative_bigram_words": negative_bigram_words
        }

    def sentiment140_unigrams(self, tokens):
        """ Sentiment 140 Unigram Lexicons features """
        sentiment140_unigrams = "../data/lexicons/Sentiment140-Lexicon-v0.1/unigrams-pmilexicon.txt.gz"
        with gzip.open(sentiment140_unigrams, 'rb') as f:
            lines = f.read().splitlines()
            lexicon_map = {}
            for l in lines:
                splits = l.decode('utf-8').split('\t')
                lexicon_map[splits[0]] = float(splits[1])

        positive_score, negative_score = 0.0, 0.0
        positive_unigram_words, negative_unigram_words = 0, 0   

        # loop through the bigrams  
        for token in tokens:
            if token in lexicon_map:
                if lexicon_map[token] >= 0:
                    positive_score += lexicon_map[token]
                    positive_unigram_words += 1
                else:
                    negative_score += lexicon_map[token]
                    negative_unigram_words += 1
        return {
            "sentiment140_positive_unigram_score": positive_score, 
            "sentiment140_negative_unigram_score": negative_score, 
            "sentiment140_postive_unigram_words": positive_unigram_words, 
            "sentiment140_negative_unigram_words": negative_unigram_words
        }

    def sentiment140_bigrams(self, tokens):
        """ Sentiment 140 Unigram Lexicons features """
        sentiment140_bigrams = "../data/lexicons/Sentiment140-Lexicon-v0.1/bigrams-pmilexicon.txt.gz"
        with gzip.open(sentiment140_bigrams, 'rb') as f:
            lines = f.read().splitlines()
            lexicon_map = {}
            for l in lines:
                splits = l.decode('utf-8').split('\t')
                lexicon_map[splits[0]] = float(splits[1])

        positive_score, negative_score = 0.0, 0.0
        positive_bigram_words, negative_bigram_words = 0, 0 

        # loop through the bigrams  
        for token in self.get_bigrams(tokens):
            if token in lexicon_map:
                if lexicon_map[token] >= 0:
                    positive_score += lexicon_map[token]
                    positive_bigram_words += 1
                else:
                    negative_score += lexicon_map[token]
                    negative_bigram_words += 1

        return {
            "sentiment140_positive_bigram_score": positive_score, 
            "sentiment140_negative_bigram_score": negative_score, 
            "sentiment140_postive_bigram_words": positive_bigram_words, 
            "sentiment140_negative_bigram_words": negative_bigram_words
        }
    
    def senti_wordnet(self, tokens):
        """ Returns features based on the SentiWordNet features """

        senti_wordnet_path = "../data/lexicons/SentiWordNet_3.0.0.txt.gz"
        with gzip.open(senti_wordnet_path, 'rb') as f:
            lines = f.read().splitlines()
            senti_wordnet_lexicon_map = defaultdict(float)

            for l in lines:
                l = l.decode('utf-8')
                if l.strip().startswith('#'):
                    continue
                splits = l.split('\t')
                # positive score - negative score
                score = float(splits[2]) - float(splits[3])
                words = splits[4].split(" ")
                # iterate through all words
                for word in words:
                    word, rank = word.split('#')
                    # scale scores according to rank
                    # more popular => less rank => high weight
                    senti_wordnet_lexicon_map[word] += (score / float(rank))

        positive_score, negative_score = 0.0, 0.0
        positive_unigram_words, negative_unigram_words = 0, 0

        # loop through the bigrams  
        for token in tokens:
            if token in senti_wordnet_lexicon_map:
                if senti_wordnet_lexicon_map[token] >= 0:
                    positive_score += senti_wordnet_lexicon_map[token]
                    positive_unigram_words += 1
                else:
                    negative_score += senti_wordnet_lexicon_map[token]
                    negative_unigram_words += 1

        return  {
            "senti_wordnet_positive_score": positive_score, 
            "senti_wordnet_negative_score": negative_score, 
            "senti_wordnet_postive_words": positive_unigram_words, 
            "senti_wordnet_negative_words": negative_unigram_words
        }

    def featurize(self, tokens):
        """ Build a feature vector for the tokens """
        features = {}
        nrc_hashtag_emotion_features = self.nrc_hashtag_emotion(tokens)
        nrc_affect_intensity_features = self.nrc_affect_intensity(tokens)
        nrc_hashtag_sentiment_lexicon_unigrams_features = self.nrc_hashtag_sentiment_lexicon_unigrams(tokens)
        nrc_hashtag_sentiment_lexicon_bigrams_features = self.nrc_hashtag_sentiment_lexicon_bigrams(tokens)
        sentiment140_unigrams_features = self.sentiment140_unigrams(tokens)
        sentiment140_bigrams_features = self.sentiment140_bigrams(tokens)
        senti_wordnet_features = self.senti_wordnet(tokens)

        features.update(nrc_hashtag_emotion_features) # 10 features
        features.update(nrc_affect_intensity_features) # 10 features
        features.update(nrc_hashtag_sentiment_lexicon_unigrams_features) # 4 features
        features.update(nrc_hashtag_sentiment_lexicon_bigrams_features) # 4 features
        features.update(sentiment140_unigrams_features) # 4 features 
        features.update(sentiment140_bigrams_features) # 4 features
        features.update(senti_wordnet_features) # 4 features

        return features

# Example of using the featurizer
def test_featurizer():
    featurizer = LexiconFeaturizer()
    features = featurizer.featurize([u'So', u'my', u'Indian', u'Uber', u'driver', u'just', u'called', u'someone', u'the', u'N', u'word', u'.', u'If', u'I', u"wasn't", u'in', u'a', u'moving', u'vehicle', u"I'd", u'have', u'jumped', u'out', u'#disgusted'])
    return features

print(test_featurizer())