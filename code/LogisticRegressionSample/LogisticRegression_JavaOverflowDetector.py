# -*- coding:utf-8 -*-
#使用Logistic Regression检测Java溢出攻击

import re
import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score


def load_one_flle(filename):
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
            print("Load file(%s)" % path)
            y.append(0)
    return x,y

#对训练异常集合的路径处理
def dirlist(path, allfile):
    filelist = os.listdir(path)

    for filename in filelist:
        filepath = path + filename
        print(filepath)
        if os.path.isdir(filepath):
            dirlist(filepath+'/', allfile)
        else:
            allfile.append(filepath)
    return allfile

#加载含有java溢出攻击的文件
def load_adfa_java_files(rootdir):
    x=[]
    y=[]
    allfile=dirlist(rootdir,[])
    for file in allfile:
        if re.match(r"../../data/ADFA-LD/Attack_Data_Master/Java_Meterpreter_\d+/UAD-Java-Meterpreter*",file):
            print("Load file(%s)" % file)
            x.append(load_one_flle(file))
            y.append(1)
    return x,y



if __name__ == '__main__':

    x1,y1=load_adfa_training_files("../../data/ADFA-LD/Training_Data_Master/")
    x2,y2=load_adfa_java_files("../../data/ADFA-LD/Attack_Data_Master/")

    x=x1+x2
    y=y1+y2
    #print(x)
    vectorizer = CountVectorizer(min_df=1)
    x=vectorizer.fit_transform(x)
    x=x.toarray()



    logreg = linear_model.LogisticRegression(C=1e5)

    score=cross_val_score(logreg, x, y, n_jobs=-1, cv=10)
    print("预测准确率为:"+str(np.mean(score)))







