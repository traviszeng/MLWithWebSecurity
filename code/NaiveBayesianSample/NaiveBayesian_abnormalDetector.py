# -*- coding:utf-8 -*-
#使用朴素贝叶斯分类器检测异常操作


import numpy as np
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import cross_val_score

N = 90

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

    scoreKNN = 0
    scoreNB = 0
    #遍历50个用户的数据计算平均准确率
    for i in range(1,51):
        user_cmd_list,dist=load_user_cmd_new("../../data/MasqueradeDat/User"+str(i))
        #print("Dist:(%s)" % dist)
        user_cmd_feature=get_user_cmd_feature_new(user_cmd_list,dist)

        #index=2 即为User3对应的label
        labels=get_label("../../data/MasqueradeDat/label.txt",i-1)
        #print(labels)
        #前5000个记录为正常操作 即前50个序列为正常操作
        y=[0]*50+labels

        x_train=user_cmd_feature[0:N]
        y_train=y[0:N]

        x_test=user_cmd_feature[N:150]
        y_test=y[N:150]

        #用KNN和朴素贝叶斯分别训练进行对比
        clfKNN = KNeighborsClassifier(n_neighbors=3)
        clfKNN.fit(x_train,y_train)
        y_predictknn=clfKNN.predict(x_test)
        score=np.mean(y_test==y_predictknn)*100
        scoreKNN +=score
        print("由第"+str(i)+"个用户数据得到的"+"KNN的准确率为：",score,'%')


        clf = GaussianNB().fit(x_train,y_train)

        y_predict = clf.predict(x_test)
        score = np.mean(y_predict==y_test)*100
        print("由第"+str(i)+"个用户数据得到的"+"NB的准确率为：",score,'%')
        scoreNB+=score

    print("KNN预测准确率为：",scoreKNN/50,'%')
    print("NB预测准确率为：",scoreNB/50,'%')
