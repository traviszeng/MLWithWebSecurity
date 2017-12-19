import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets


iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

h = 0.02

logreg = linear_model.LogisticRegression(C=1e5)


logreg.fit(X, Y)


#按两个特征的最大值最小值生成步长为0.02的等差数列特征
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
#生成171*231的特征组
x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
Z = logreg.predict(np.c_[x1.ravel(), x2.ravel()])


Z = Z.reshape(x1.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(x1, x2, Z, cmap=plt.cm.Paired)


plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
plt.xticks(())
plt.yticks(())

plt.show()

