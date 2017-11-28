# -*- coding:utf-8 -*-

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import numpy as np


import pickle
import gzip


def load_data():

    with gzip.open('../../data/MNIST/mnist.pkl.gz') as fp:
            training_data, valid_data, test_data = pickle.load(fp)

    return training_data, valid_data, test_data


if __name__ == '__main__':
    training_data, valid_data, test_data=load_data()
    x1,y1=training_data
    x2,y2=test_data
    clf = GaussianNB()
    clf.fit(x1, y1)
    score = cross_val_score(clf, x2, y2, scoring="accuracy")
    print("预测准确率为",np.mean(score))