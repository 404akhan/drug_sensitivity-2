# so you have a mask, mask2, seed for division, threshold
# apply different methods and see which better performance

from helpers import *
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

gl_X, y = get_X_y()

print gl_X.shape

mask = get_mask(3)
X = gl_X[:, mask]

mask2 = get_new_mask(X, y, model=linear_model.Lasso(alpha=1e-2))
X = X[:, mask2]

models = {}
models['LinearRegression'] = linear_model.LinearRegression()
models['Lasso'] = linear_model.Lasso(alpha=1e-2)
models['Ridge'] = linear_model.Ridge(alpha=1)
models['ElasticNet'] = linear_model.ElasticNet(alpha=0.4, l1_ratio=0.55)
models['Lars'] = linear_model.Lars(n_nonzero_coefs=1)
models['LassoLars'] = linear_model.LassoLars(alpha=0.1)
models['BayesianRidge'] = linear_model.BayesianRidge()
models['KernelRidge'] = kernel_ridge.KernelRidge(alpha=0.5)
models['GradientBoostingRegressor'] = GradientBoostingRegressor()
models['SVR'] = svm.SVR()
models['RandomForestRegressor'] = RandomForestRegressor()
models['ExtraTreesRegressor'] = ExtraTreesRegressor()
models['DecisionTreeRegressor'] = DecisionTreeRegressor()
models['MLPRegressor'] = MLPRegressor(max_iter=10*1000, activation='logistic', hidden_layer_sizes=(100,))

for k, v in models.iteritems():
    arr_error = []
    sum_err = 0.
    sum_err_tr = 0.
    sum_baseline = 0.
    for i in [0, 1, 2]:
        X_train, X_test, y_train, y_test = get_tr_cv(X, y, i, 7174)
        error, error_tr, baseline = error_method_any(v, X_train, X_test, y_train, y_test)
        arr_error.append(error)

        sum_err += np.mean(error)
        sum_err_tr += np.mean(error_tr)
        sum_baseline += np.mean(baseline)
    print 'method: ' + k + ', error: ' + str(sum_err/3), 'error_tr: ' + str(sum_err_tr/3), 'sum_baseline: ' + str(sum_baseline/3)


