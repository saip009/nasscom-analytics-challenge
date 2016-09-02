import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_excel("Health_care_Dataset_for_probelm.xlsx", sheetname='Training Data', skiprows=5)

data.drop(["Patient_ID"],axis = 1, inplace = True)
labels = pd.DataFrame(data["Lung_Cancer"])
data.drop(["Lung_Cancer"],axis = 1, inplace = True)

data = pd.get_dummies(data, columns=["Factor1","Factor5","DiseaseHis1","DiseaseHis2","DiseaseHis3","DiseaseHis4","DiseaseHis5","DiseaseHis7","DiseaseStage1","DiseaseStage2","Disease1","Disease1Treat","Disease2","Disease3","Disease4","Disease4Treat","Disease5","Disease5Treat","Disease6","Disease6Treat","Disease7"])

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=33)

# from sklearn import feature_selection
# fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=100)
# x_train_fs = fs.fit_transform(x_train, y_train)


from sklearn.linear_model import SGDClassifier
# x = 0.001
# while True:
#     clf = SGDClassifier(loss='log', alpha=0.2, epsilon=x)
#     clf.fit(x_train, y_train)
#     pred = clf.predict(x_test)
#
#     from sklearn.metrics import accuracy_score
#     accuracy = accuracy_score(y_test,pred)
#     print str(x) + '\t\t' + str(accuracy)
#
#     x += 0.005
#
#     if x >= 1:
#         break

# clf = SGDClassifier(loss='log', alpha=0.2, n_jobs=2, )
# clf.fit(x_train, y_train)
# pred = clf.predict(x_test)

# GRIDSEARCH
from sklearn import grid_search

clf = SGDClassifier(alpha=0.02, loss='modified_huber', n_iter=1000)
# parameters = {
#     'alpha': [x*0.01 for x in range(0,200)],
#     'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
#     'penalty': [ 'none', 'l2', 'l1', 'elasticnet'],
#     'n_iter': [x for x in range(0,2000,10)]
#
# }
# clf = grid_search.GridSearchCV(sgd, parameters)

clf.fit(x_train, y_train)
# x_test = fs.transform(x_test)
y_pred = clf.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print str(accuracy)

