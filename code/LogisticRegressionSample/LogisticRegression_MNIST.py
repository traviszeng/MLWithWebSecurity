# -*- coding:utf-8 -*-

import re
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model, datasets
from sklearn.model_selection import cross_val_score
from tensorflow.examples.tutorials.mnist import input_data




def label2Vec(yy1):
    y1 = []
    for yy in yy1:
        flag = 0
        for yyy in yy:
            if yyy==1:
                y1.append(flag)
            else:
                flag+=1
    return y1

def load_data():
    mnist = input_data.read_data_sets("../../data/MNIST", one_hot=True)
    return mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels


if __name__ == '__main__':
    x1,y1,x2,y2 = load_data()

    y1 = label2Vec(y1)
    y2 = label2Vec(y2)

    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(x1, y1)
    score = cross_val_score(logreg, x2, y2, scoring="accuracy")
    print("预测准确率为", np.mean(score))





