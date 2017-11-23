from gensim import models
import numpy as np


class EmbeddingFeaturizer(object):
	def __init__(self, glove=False):
		if glove:
		  self.glove_model = self.load_glove_embedding()

    def load_glove_embedding(self):
        """ Load the glove embedding using gensim's word2vec api """
        glove_embedding_location = "./data/word2vec/w2v.twitter.txt"
        print("Loading glove embeddings...")
        glove_model = models.KeyedVectors.load_word2vec_format(
          glove_embedding_location, binary=False)
        return glove_model

    def glove_embeddings_for_tweet(self, tokens):
      """ Taking the average of the word embeddings in a tweet """
      sum_vec = np.zeros(shape=(self.glove_model.vector_size,))
      for token in tokens:
           if token in self.glove_model:
              sum_vec = sum_vec + self.glove_model[token]
          else:
           if token.startswith("#"):
              without_hashtag = token[1:]
              if without_hashtag in self.glove_model:
                sum_vec = sum_vec + self.glove_model[without_hashtag]

        denom = len(tokens)
        sum_vec = sum_vec / denom
        return sum_vec

    def glove_embedding_for_word(self, word):
      """ Return the glove embedding for a single word """
      return self.glove_model[word]



