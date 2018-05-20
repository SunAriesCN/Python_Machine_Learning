from sklearn.ensemble import RandomForestClassifier
from plot_decision_regions_sklearn import plot_decision_regions
from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from plot_decision_regions_sklearn import plt
from plot_decision_regions_sklearn import plot_decision_regions

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105,150))

plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()
