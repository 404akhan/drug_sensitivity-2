# submit

from helpers import *
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import sys

all_X, y = get_X_y()

mask = get_mask(3)
X = all_X[:, mask]

def get_new_mask():
    clf = linear_model.Lasso(alpha=1e-2)
    rfecv = RFECV(clf, step=1, cv=3)
    selector = rfecv.fit(X, y)
    return selector.support_

mask2 = get_new_mask()
X = X[:, mask2]

lr = linear_model.LinearRegression()
lr.fit(X, y)

########
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

X_test = data[:, 3:][:, mask][:, mask2]
y_test = data[:, 2]
ids = data[:, 0]

print X_test
print y_test
print ids

predict = lr.predict(X_test)

print predict

print '"ID","IC50"'
for i in range(len(ids)):
    print '%d,%.6f' % (ids[i], predict[i])
