""" Implementation of various evaluation metrics """
from sklearn.metrics import mean_squared_error
import scipy
import math


def rmse(y_true, y_pred):
    """ Root mean squared error """
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mse(y_true, y_pred):
    """ Mean Squared Error """
    return mean_squared_error(y_true, y_pred)

def pearson_correlation(x, y):
	""" Caculates the pearson correlation between two datasets """
	return scipy.stats.pearsonr(x, y)[0]