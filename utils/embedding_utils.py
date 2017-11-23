""" Script for processing embeddings """

from gensim.models import Word2Vec 
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format("../data/word2vec/w2v.twitter.txt")

print model
