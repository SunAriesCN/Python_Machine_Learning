from plot_decision_regions_sklearn import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from plot_decision_regions_sklearn import plt
from plot_decision_regions_sklearn import plot_decision_regions

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)

from sklearn.tree import export_graphviz
tree.fit(X_train, y_train)
export_graphviz(tree, out_file='tree.dot', feature_names=['petal length','petal width'])
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105,150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()
