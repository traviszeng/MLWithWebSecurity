# -*- coding:utf-8 -*-

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import numpy as np
import tensorflow as tf
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

if __name__ == '__main__':
    mnist = input_data.read_data_sets("../../data/MNIST",one_hot = True)
    x1,yy1=mnist.train.images,mnist.train.labels
    x2,yy2=mnist.test.images,mnist.test.labels

    y1 = label2Vec(yy1)
    y2 = label2Vec(yy2)
                
    clf = GaussianNB()
    clf.fit(x1, y1)
    score = cross_val_score(clf, x2, y2, scoring="accuracy")
    print("预测准确率为",np.mean(score))
