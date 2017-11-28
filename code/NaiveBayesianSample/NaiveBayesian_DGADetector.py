# -*- coding:utf-8 -*-

import numpy as np
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB

#处理域名的最小长度
MIN_LEN=10

#加载白样本网址
def load_alexa(filename):
    domain_list = []
    csv_reader = csv.reader(open(filename))
    for row in csv_reader:
        domain = row[1]
        if len(domain)>=MIN_LEN:
            domain_list.append(domain)
    return domain_list


def load_dga(filename):
    domain_list=[]
    #xsxqeadsbgvpdke.co.uk,Domain used by Cryptolocker - Flashback DGA for 13 Apr 2017,2017-04-13,
    # http://osint.bambenekconsulting.com/manual/cl.txt
    with open(filename) as f:
        for line in f:
            #数据项第一个为网址
            domain=line.split(",")[0]
            #如果长度大于最小长度
            if len(domain)>= MIN_LEN:
                domain_list.append(domain)
    return  domain_list


def nb_dga():
    #加载alexa前1000的域名作为白样本，标记为0
    alexa_domain_list = load_alexa("../../data/DGA_data/top-1000.csv")
    #分别加载cryptolocker和post-tovar-goz家族的DGA域名标记为2和3
    cryptolocker_domain_list = load_dga("../../data/DGA_data/dga-cryptolocke-1000.txt")
    post_tovar_goz_domain_list = load_dga("../../data/DGA_data/dga-post-tovar-goz-1000.txt")

    x_domain_list=np.concatenate((alexa_domain_list, cryptolocker_domain_list,post_tovar_goz_domain_list))

    y1=[0]*len(alexa_domain_list)
    y2=[1]*len(cryptolocker_domain_list)
    y3=[2]*len(post_tovar_goz_domain_list)

    y=np.concatenate((y1, y2,y3))


    #2-gram分割域名，切割单元为字符，以整个数据集合的2-gram结果作为词汇表并进行映射，得到特征化的向量
    cv = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",
                                          token_pattern=r"\w", min_df=1)
    x= cv.fit_transform(x_domain_list).toarray()


    clf = GaussianNB()
    score  = cross_val_score(clf, x, y, n_jobs=-1, cv=3)
    print("准确率为:",np.mean(score))
    print(score)

if __name__ == '__main__':
    nb_dga()



