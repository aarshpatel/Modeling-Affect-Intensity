class EmbeddingFeaturizer(object):
	def __init__(self):
		pass
	def glove_embeddings(self, tokens, dim):
		pass
	def word2vec_embeddings(self, tokens, dim):
		pass
	def emoji2vec_embeddings(self, tokens):
		pass
	def featurizer(tokens, dim):
		""" Returns a avg embedding of the tweet as the features for the tweet """ 

		features = []
		glove_embedding_features = self.glove_embeddings(tokens, dim)
		word2vec_embeddings_features = self.word2vec_embeddings(tokens, dim)
		emoji2vec_embeddings_features = self.emoji2vec_embeddings(tokens)
		features.extend(glove_embeddings_features)
				.extend(emoji2vec_embeddings_features)	
				.extend(word2vec_embeddings_features)
		return features		


