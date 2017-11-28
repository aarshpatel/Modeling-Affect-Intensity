from gensim import models
import numpy as np


class EmbeddingFeaturizer(object):
    def __init__(self, glove=False, w2v=False, emoji=False):

        if glove:
            self.glove_model = self.load_glove_embeddings()

        if w2v:
            self.w2v_model = self.load_w2v_embeddings()

        if emoji:
            self.emoji_model = self.load_emoji_embeddings()


    def load_glove_embeddings(self):
        """ Load the glove embedding using gensim's word2vec api """
        glove_embedding_location = "./data/word2vec/w2v.twitter.txt"
        print("Loading glove embeddings...")
        glove_model = models.KeyedVectors.load_word2vec_format(glove_embedding_location, binary=False)
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

    def load_w2v_embeddings(self):
        w2v_embedding_path = "./data/word2vec/word2vec-googlenews-300.bin"
        print("Loading w2v pretained embeddings...")
        w2v_model = models.KeyedVectors.load_word2vec_format(w2v_embedding_path, binary=True)
        return w2v_model

    def w2v_embeddings_for_tweets(self, tokens):
        """ Taking the average of the word embeddings in a tweet """
        sum_vec = np.zeros(shape=(self.w2v_model.vector_size,))
        for token in tokens:
            if token in self.w2v_model:
                sum_vec = sum_vec + self.w2v_model[token]
            else:
                if token.startswith("#"):
                    without_hashtag = token[1:]
                    if without_hashtag in self.w2v_model:
                        sum_vec = sum_vec + self.w2v_model[without_hashtag]

        denom = len(tokens)
        sum_vec = sum_vec / denom
        return sum_vec

    def load_emoji_embeddings(self):
        """ Load the emoji embeddings """
        emoji_embeddings_path = "./data/word2vec/emoji2vec.bin"
        print("Loading Emoji Embeddings")
        model = models.KeyedVectors.load_word2vec_format(emoji_embeddings_path, binary=True)
        return model

    def emoji_embeddings_for_tweets(self, tokens):
        """ Take the average of emoji embeddings """
        sum_vec = np.zeros(shape=(self.emoji_model.vector_size,))
        emojis = 0
        for token in tokens:
            try:
                if token.decode('utf-8') in self.emoji_model:
                    emojis += 1
                    sum_vec = sum_vec + self.emoji_model[token.decode('utf-8')]
            except:
                pass
        sum_vec = sum_vec / emojis
        return sum_vec
