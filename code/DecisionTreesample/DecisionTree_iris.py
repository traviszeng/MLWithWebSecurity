#决策树的helloworld 使用决策树对iris数据集进行分类

from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus

iris = load_iris()


clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)


dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")
