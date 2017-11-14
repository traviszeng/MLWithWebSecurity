# -*- coding:utf-8 -*-

import re
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np


def load_kdd99(filename):
    x=[]
    with open(filename) as f:
        for line in f:
            line=line.strip('\n')
            line=line.split(',')
            x.append(line)
    return x

def get_rootkit2andNormal(x):
    #保存与TCP连接相关的内容特征
    tcpFeatures=[]
    w=[]
    label=[]
    for x1 in x:
        #筛选标记为rootkit或者normal的并且协议为telnet的数据
        if ( x1[41] in ['rootkit.','normal.'] ) and ( x1[2] == 'telnet' ):
            if x1[41] == 'rootkit.':
                label.append(1)
            else:
                label.append(0)

            #保存TCP连接相关特征
            x1 = x1[9:21]
            tcpFeatures.append(x1)

    #将rootkit相关特征转为float
    for x1 in tcpFeatures :
        v1=[]
        for x2 in x1:
            v1.append(float(x2))
        w.append(v1)
    return w,label

if __name__ == '__main__':
    v=load_kdd99("../data/kddcup99/corrected")
    x,y=get_rootkit2andNormal(v)
    clf = KNeighborsClassifier(n_neighbors=3)
    #十折交叉验证
    score = cross_val_score(clf, x, y, n_jobs=-1, cv=10)
    print(score)
    print(np.mean(score))




