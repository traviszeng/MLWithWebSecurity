from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB

#高斯贝叶斯分类器
gnb = GaussianNB()
#拟合数据
y_pred = gnb.fit(iris.data,iris.target).predict(iris.data)

print("Ratio of correctness:",1-(iris.target!=y_pred).sum()/iris.data.shape[0])
