""" Implementation of various evaluation metrics """
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
import scipy
import math


def rmse(y_true, y_pred):
    """ Root mean squared error """
    return math.sqrt(mean_squared_error(y_true, y_pred))

def mse(y_true, y_pred):
    """ Mean Squared Error """
    return mean_squared_error(y_true, y_pred)

def pearson_correlation(y_true, y_pred):
	""" Caculates the pearson correlation between two datasets """
	return scipy.stats.pearsonr(y_true, y_pred)[0]

# scikit-learn scorer funcs
rmse_scorer = make_scorer(rmse, greater_is_better=False)
mse_scorer = make_scorer(mse, greater_is_better=False)
pearson_scorer = make_scorer(pearson_correlation, greater_is_better=True)
