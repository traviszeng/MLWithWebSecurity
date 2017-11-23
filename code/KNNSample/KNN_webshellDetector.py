# -*- coding:utf-8 -*-
#使用KNN算法检测webshell

import re
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

#加载文件中的内容
def load_one_flle(filename):
    x=[]
    with open(filename) as f:
        line=f.readline()
        line=line.strip('\n')
    return line

#找到训练集文件
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

#对训练异常集合的路径处理
def dirlist(path, allfile):
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = path+filename
        #print(filepath)
        if os.path.isdir(filepath):
            dirlist(filepath+'/', allfile)
        else:
            allfile.append(filepath)
    return allfile

#加载含有webshell操作的文件
def load_adfa_webshell_files(rootdir):
    x=[]
    y=[]
    allfile=dirlist(rootdir,[])
    for file in allfile:
        if re.match(r"../../data/ADFA-LD/Attack_Data_Master/Web_Shell_\d+/UAD-W*",file):
            x.append(load_one_flle(file))
            y.append(1)
    return x,y



if __name__ == '__main__':

    x1,y1=load_adfa_training_files("../../data/ADFA-LD/Training_Data_Master/")
    x2,y2=load_adfa_webshell_files("../../data/ADFA-LD/Attack_Data_Master/")

    x=x1+x2
    y=y1+y2
    #print x
    vectorizer = CountVectorizer(min_df=1)
    x=vectorizer.fit_transform(x)
    x=x.toarray()
    #print y
    clf = KNeighborsClassifier(n_neighbors=3)
    scores=cross_val_score(clf, x, y, n_jobs=-1, cv=10)
    print(scores)
    print(np.mean(scores))
