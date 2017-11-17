""" Implementation of various metrics """

from sklearn.metrics import mean_squared_error
import math

def rmse(y_true, y_pred):
	""" Root mean squared error """
	return math.sqrt(mean_squared_error(y_true, y_pred))

def mse(y_true, y_pred):
	""" Mean Squared Error """
	return mean_squared_error(y_true, y_pred)
	

