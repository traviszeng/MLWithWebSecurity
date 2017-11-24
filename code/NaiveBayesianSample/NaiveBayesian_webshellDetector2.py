# _*_coding:utf-8_*_
#朴素贝叶斯检测webshell

import os
import sys
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB

#针对函数调用建立特征
token_pattern = r'\b\w+\b\(|\'\w+\''

def load_file(filepath):
    t = ""
    try:
        with open(filepath,'r', encoding='UTF-8') as f:
            for line in f:
                line = line.strip('\n')
                t+=line
    except UnicodeDecodeError:
        pass
    return t


def load_files(filepath):
    files_list = []
    for r,d,files in os.walk(filepath):
        for file in files:
            if file.endswith('.php'):
                file_path = filepath + file
                print("Load %s" % file_path)
                t = load_file(file_path)
                files_list.append(t)

    return files_list

if __name__ =='__main__':

    webshell_bigram_vectorizer = CountVectorizer(ngram_range=(1,1), decode_error="ignore",
                                        token_pattern = token_pattern,min_df=1)
    
    webshell_files_list=load_files("../../data/PHP-WEBSHELL/xiaoma/")
    x1=webshell_bigram_vectorizer.fit_transform(webshell_files_list).toarray()
    y1=[1]*len(x1)
    #生成黑样本的vocabulary
    vocabulary=webshell_bigram_vectorizer.vocabulary_

    #使用黑样本生成的词汇表vocabulary将白样本特征化，注意设置CountVectorizer函数的vocabulary 这样才能以黑样本的生成的词汇表来进行向量化
    wp_bigram_vectorizer = CountVectorizer(ngram_range=(1,1), decode_error="ignore",
                                        token_pattern = token_pattern,min_df=1,vocabulary=vocabulary)
    wp_files_list=load_files("../../data/wordpress/")
    x2=wp_bigram_vectorizer.fit_transform(wp_files_list).toarray()
    y2=[0]*len(x2)

    x=np.concatenate((x1,x2))
    y=np.concatenate((y1,y2))

    clf = GaussianNB()
    print(vocabulary)
    score = cross_val_score(clf,x,y,n_jobs = -1,cv=3)
    print(score)
    print(np.mean(score))

