import sys
import urllib
import re
from hmmlearn import hmm
import numpy as np
from sklearn.externals import joblib
#import HTMLParser python2
from html.parser import HTMLParser
import nltk


#处理参数值的最小长度
MIN_LEN=6

#状态个数
N=10
#最大似然概率阈值
T=-200
#字母
#数字 1
#<>,:"'
#其他字符2
SEN=['<','>',',',':','\'','/',';','"','{','}','(',')']

def ischeck(str):
    if re.match(r'^(http)',str):
        return False
    for i, c in enumerate(str):
        if ord(c) > 127 or ord(c) < 31:
            return False
        if c in SEN:
            return True
        #排除中文干扰 只处理127以内的字符


    return False

def etl(str):
    vers=[]
    for i, c in enumerate(str):
        c=c.lower()
        if   ord(c) >= ord('a') and  ord(c) <= ord('z'):
            vers.append([ord(c)])
        elif ord(c) >= ord('0') and  ord(c) <= ord('9'):
            vers.append([1])
        elif c in SEN:
            vers.append([ord(c)])
        else:
            vers.append([2])

    #print vers
    return np.array(vers)

def do_str(line):
    words=nltk.word_tokenize(line)
    print(words)

def main(filename):
    X = [[0]]
    X_lens = [1]
    with open(filename) as f:
        for line in f:
            line=line.strip('\n')
            #url解码
            line=urllib.parse.unquote(line)
            #处理html转义字符
            line=HTMLParser().unescape(line)
            if len(line) >= MIN_LEN:
                print("Learning xss query param:(%s)" % line)
                do_str(line)

            #X=np.concatenate( [X,vers])
            #X_lens.append(len(vers))


    #print X
    #remodel = hmm.GaussianHMM(n_components=N, covariance_type="full", n_iter=100)
    #remodel.fit(X,X_lens)
    #joblib.dump(remodel, "xss-train.pkl")

    #return remodel

def test(remodel,filename):
    with open(filename) as f:
        for line in f:
            # 切割参数
            result = urlparse.urlparse(line)
            # url解码
            query = urllib.parse.unquote(result.query)
            params = urlparse.parse_qsl(query, True)

            for k, v in params:

                if ischeck(v) and len(v) >=N :
                    vers = etl(v)
                    pro = remodel.score(vers)
                    #print  "CHK SCORE:(%d) QUREY_PARAM:(%s) XSS_URL:(%s) " % (pro, v, line)
                    if pro >= T:
                        print("SCORE:(%d) QUREY_PARAM:(%s) XSS_URL:(%s) " % (pro,v,line))
                        #print line



if __name__ == '__main__':
    #remodel=main(sys.argv[1])
    #test(remodel,sys.argv[2])
    #nltk.download()
    main('../../data/XSSdata/good-xss-10000.txt')
