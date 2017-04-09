import numpy as np
import pandas as pd
import random
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV

def get_corrcoef_1d(x, y):
    x_c, y_c = [], []
    for i in x: x_c.append(i)
    for i in y: y_c.append(i)
    return np.corrcoef(x_c, y_c)[0][1]

def get_mask(threshold = 8):
    df = pd.read_csv("./onedrug_train.csv")
    table = np.array(df)

    nan_index = []
    good_index = []
    for i in range(len(table)):
        if np.isnan(table[i, 2]):
            nan_index.append(i)
        else:
            good_index.append(i)

    data = table[good_index, :]
    X = data[:, 3:]

    mask = np.sum(X, axis=0) > threshold

    return mask

def get_X_y():
    df = pd.read_csv("./onedrug_train.csv")

    table = np.array(df)

    nan_index = []
    good_index = []
    for i in range(len(table)):
        if np.isnan(table[i, 2]):
            nan_index.append(i)
        else:
            good_index.append(i)

    data = table[good_index, :]

    X = data[:, 3:]
    y = data[:, 2]

    return X, y

def get_X_test():
    df = pd.read_csv("./onedrug_train.csv")

    table = np.array(df)

    nan_index = []
    good_index = []
    for i in range(len(table)):
        if np.isnan(table[i, 2]):
            nan_index.append(i)
        else:
            good_index.append(i)

    data = table[nan_index, :]

    X = data[:, 3:]

    return X

def get_tr_cv(X, y, i, seed=7174):
    random.seed(seed)
    shuf_arr = range(0, len(y))
    random.shuffle(shuf_arr)
    test_size = int(len(y) * 0.3333)

    sh_y = y[shuf_arr]
    sh_X = X[shuf_arr, :]

    arr_x = []
    arr_x.append(sh_X[:test_size, :])
    arr_x.append(sh_X[test_size:2 * test_size + 1, :])
    arr_x.append(sh_X[test_size * 2 + 1:, :])

    arr_y = []
    arr_y.append(sh_y[:test_size])
    arr_y.append(sh_y[test_size:2 * test_size + 1])
    arr_y.append(sh_y[test_size * 2 + 1:])

    xs_test = arr_x[i]
    ys_test = arr_y[i]
    i2, i3 = (i + 1) % 3, (i + 2) % 3
    xs_train = np.vstack((arr_x[i2], arr_x[i3]))
    ys_train = np.concatenate((arr_y[i2], arr_y[i3]))

    return xs_train, xs_test, ys_train, ys_test

def get_tr_cv_2(X, y, i):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42)

    if i == 0: return X_train, X_test, y_train, y_test
    if i == 1: return X_test, X_train, y_test, y_train

def error_method_linear(X_train, X_test, y_train, y_test):
    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)

    predict = lr.predict(X_test)
    error = np.mean(np.square(predict - y_test))
    baseline = np.mean(np.square(np.mean(y_test) - y_test))

    predict_tr = lr.predict(X_train)
    error_tr = np.mean(np.square(predict_tr - y_train))

    return error, error_tr, baseline

def error_method_any(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)

    predict = clf.predict(X_test)
    error = np.mean(np.square(predict - y_test))
    baseline = np.mean(np.square(np.mean(y_test) - y_test))

    predict_tr = clf.predict(X_train)
    error_tr = np.mean(np.square(predict_tr - y_train))

    return error, error_tr, baseline

def get_new_mask(X, y, model=LinearRegression()):
    clf = model
    rfecv = RFECV(clf, step=1, cv=3)
    selector = rfecv.fit(X, y)
    return selector.support_

def get_new_mask_2(X, y):
    clf = LinearRegression()
    rfecv = RFECV(clf, step=1, cv=2)
    selector = rfecv.fit(X, y)
    return selector.support_

def get_X_y_v2():
    sX, y = get_X_y()

    X = np.zeros(sX.shape)
    for i in range(sX.shape[0]):
        for j in range(sX.shape[1]):
            X[i][j] = sX[i][j]
    return X, y