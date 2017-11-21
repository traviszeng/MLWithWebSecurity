# -*- coding:utf-8 -*-
#使用决策树算法检测FTP暴力破解

import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
import os
from sklearn import tree
import pydotplus
import numpy as np


def load_one_flle(filename):
    x=[]
    with open(filename) as f:
        line=f.readline()
        line=line.strip('\n')
    return line

def load_adfa_training_files(rootdir):
    x=[]
    y=[]
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            x.append(load_one_flle(path))
            y.append(0)
    return x,y

def dirlist(path, allfile):
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = path+filename
        if os.path.isdir(filepath):
            #处理路径异常
            dirlist(filepath+'/', allfile)
        else:
            allfile.append(filepath)
    return allfile

def load_adfa_hydra_ftp_files(rootdir):
    x=[]
    y=[]
    allfile=dirlist(rootdir,[])
    for file in allfile:
        #正则表达式匹配hydra异常ftp文件
        if re.match(r"../../data/ADFA-LD/Attack_Data_Master/Hydra_FTP_\d+/UAD-Hydra-FTP*",file):
            x.append(load_one_flle(file))
            y.append(1)
    return x,y



if __name__ == '__main__':
    x1,y1=load_adfa_training_files("../../data/ADFA-LD/Training_Data_Master/")
    x2,y2=load_adfa_hydra_ftp_files("../../data/ADFA-LD/Attack_Data_Master/")

    x=x1+x2
    y=y1+y2
    vectorizer = CountVectorizer(min_df=1)
    x=vectorizer.fit_transform(x)
    x=x.toarray()
    clf = tree.DecisionTreeClassifier()
    score = cross_val_score(clf, x, y, n_jobs=-1, cv=10)
    print(score)
    print('平均正确率为：',np.mean(score))


    clf = clf.fit(x, y)
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("ftpDetector_decisionTree.pdf")
