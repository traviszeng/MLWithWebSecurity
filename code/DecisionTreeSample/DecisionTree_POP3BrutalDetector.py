#使用决策树算法检测POP3暴力破解
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
import os
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus

#加载kdd数据集
def load_kdd99(filename):
	X=[]
	with open(filename) as f:
		for line in f:
			line = line.strip('\n')
			line = line.split(',')
			X.append(line)
	return X

#找到训练数据集
def get_guess_passwdandNormal(x):
    v=[]
    features=[]
    targets=[]
    #找到标记为guess-passwd和normal且是POP3协议的数据
    for x1 in x:
        if ( x1[41] in ['guess_passwd.','normal.'] ) and ( x1[2] == 'pop_3' ):
            if x1[41] == 'guess_passwd.':
                targets.append(1)
            else:
                targets.append(0)
	    #挑选与POP3密码破解相关的网络特征和TCP协议内容的特征作为样本特征
            x1 = [x1[0]] + x1[4:8]+x1[22:30]
            v.append(x1)

    for x1 in v :
        v1=[]
        for x2 in x1:
            v1.append(float(x2))
        features.append(v1)
    return features,targets

if __name__ == '__main__':
    v=load_kdd99("../../data/kddcup99/corrected")
    x,y=get_guess_passwdandNormal(v)
    clf = tree.DecisionTreeClassifier()
    print(cross_val_score(clf, x, y, n_jobs=-1, cv=10))

    clf = clf.fit(x, y)
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("POP3Detector.pdf")
