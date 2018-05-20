
# coding: utf-8

# In[11]:

from sklearn.ensemble import RandomForestClassifier
from PatitionDatasetInTrainAndTestSet import X_train_std, y_train, X_test_std, y_test, df_wine, X_train
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')


# In[9]:

feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000,random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f+1, 30, feat_labels[f], importances[indices[f]]))


# In[12]:

plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]),importances[indices],color='lightblue',align='center')
plt.xticks(range(X_train.shape[1]),feat_labels,rotation=90)
plt.xlim([-1,X_train.shape[1]])
plt.tight_layout()
plt.show()


# In[13]:

X_selected = forest.transform(X_train, threshold=0.15)
print(X_selected.shape)

