# -*- coding:utf-8 -*-

import sys
import urllib
import re
from hmmlearn import hmm
import numpy as np
from sklearn.externals import joblib
import nltk
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import cross_val_score


#测试样本数
N=90


def load_user_cmd(filename):
    cmd_list=[]
    dist_max=[]
    dist_min=[]
    dist=[]
    with open(filename) as f:
        i=0
        x=[]
        for line in f:
            line=line.strip('\n')
            x.append(line)
            dist.append(line)
            i+=1
            #每100条命令组成一个操作序列
            if i == 100:
                cmd_list.append(x)
                x=[]
                i=0

    #统计最频繁使用的前50个命令和最少使用的50个命令
    fdist =list(FreqDist(dist).keys())
    dist_max=set(fdist[0:50])
    dist_min = set(fdist[-50:])
    return cmd_list,dist_max,dist_min

#特征化用户使用习惯
def get_user_cmd_feature(user_cmd_list,dist_max,dist_min):
    user_cmd_feature=[]
    for cmd_block in user_cmd_list:
        #以100个命令为统计单元，作为一个操作序列，去重后的操作命令个数作为特征
        #将list转为set去重
        f1=len(set(cmd_block))
        #FreqDist转为统计字典转化为命令:出现次数的形式
        fdist = list(FreqDist(cmd_block).keys())
        #最频繁使用的10个命令
        f2=fdist[0:10]
        #最少使用的10个命令
        f3 = fdist[-10:]
        f2 = len(set(f2) & set(dist_max))
        f3 = len(set(f3) & set(dist_min))
        #返回该统计单元中和总的统计的最频繁使用的前50个命令和最不常使用的50个命令的重合程度
        #f1:统计单元中出现的命令类型数量
        #f2:统计单元中最常使用的10个命令和总的最常使用的命令的重合程度
        #f3:统计单元中最不常使用的10个命令和总的最不常使用的命令的重合程度
        x=[f1,f2,f3]
        user_cmd_feature.append(x)
    return user_cmd_feature

def load_user_cmd_new(filename):
    cmd_list=[]
    dist=[]
    with open(filename) as f:
        i=0
        x=[]
        for line in f:
            line=line.strip('\n')
            x.append(line)
            dist.append(line)
            i+=1
            if i == 100:
                cmd_list.append(x)
                x=[]
                i=0

    fdist = list(FreqDist(dist).keys())
    return cmd_list,fdist

def get_user_cmd_feature_new(user_cmd_list,dist):
    user_cmd_feature=[]

    for cmd_list in user_cmd_list:
        v=[0]*len(dist)
        for i in range(0,len(dist)):
            if dist[i] in cmd_list:
                v[i]+=1
        user_cmd_feature.append(v)

    return user_cmd_feature

def get_label(filename,index=0):
    x=[]
    with open(filename) as f:
        for line in f:
            line=line.strip('\n')
            x.append( int(line.split()[index]))
    return x

if __name__ == '__main__':
    
    #user_cmd_list,user_cmd_dist_max,user_cmd_dist_min=load_user_cmd("../data/MasqueradeDat/User3")
    #user_cmd_feature=get_user_cmd_feature(user_cmd_list,user_cmd_dist_max,user_cmd_dist_min)
    user_cmd_list,dist=load_user_cmd_new("../../data/MasqueradeDat/User3")
    print("Dist:(%s)" % dist)
    user_cmd_feature=get_user_cmd_feature_new(user_cmd_list,dist)

    #index=2 即为User3对应的label
    labels=get_label("../../data/MasqueradeDat/label.txt",2)
    #前5000个记录为正常操作 即前50个序列为正常操作
    y=[0]*50+labels

    x_train=user_cmd_feature[0:N]
    y_train=y[0:N]

    x_test=user_cmd_feature[N:150]
    y_test=y[N:150]

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train, y_train)
    y_predict=neigh.predict(x_test)

    #score=np.mean(y_test==y_predict)*100
    
    score=cross_val_score(neigh,user_cmd_feature,y,n_jobs = -1,cv=10)

    #print y
    #print y_train
    print(y_test)
    print(y_predict)
    print(score)

    print(classification_report(y_test, y_predict))

    print(metrics.confusion_matrix(y_test, y_predict))
