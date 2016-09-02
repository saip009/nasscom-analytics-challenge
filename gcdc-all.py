import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_excel("Health_care_Dataset_for_probelm.xlsx", sheetname='Training Data', skiprows=5)
data_test = pd.read_excel("Health_care_Dataset_for_probelm.xlsx", sheetname='Evaluation Data', skiprows=5)
data.drop(["Patient_ID"],axis = 1, inplace = True)
data_test.drop(["Patient_ID"],axis = 1, inplace = True)
labels = pd.DataFrame(data["Lung_Cancer"])
data_test.drop(["Lung_Cancer"],axis = 1, inplace = True)
data.drop(["Lung_Cancer"],axis = 1, inplace = True)

# print data.columns

# print data.info()

# data = data[["Factor1","Factor3","Factor4","Factor6","DiseaseHis1","DiseaseHis1Times","DiseaseHis2","DiseaseHis2Times","DiseaseHis3","DiseaseHis3Times","DiseaseHis4","DiseaseHis6","DiseaseHis7","DiseaseStage2","LungFunct4","LungFunct5","LungFunct6","LungFunct10","LungFunct14","LungFunct15","LungFunct17","LungFunct19","LungFunct20","Disease1","Disease1Treat","Disease2","Disease2Times","Disease3","Disease3Times","Disease4","Disease4Treat","Disease5","Disease5Treat","Disease6","Disease7","Ques1","Ques2","Ques3","Ques5","Smoke1","Smoke2","Smoke3","Smoke4"]]
# data_test = data_test[["Factor1","Factor3","Factor4","Factor6","DiseaseHis1","DiseaseHis1Times","DiseaseHis2","DiseaseHis2Times","DiseaseHis3","DiseaseHis3Times","DiseaseHis4","DiseaseHis6","DiseaseHis7","DiseaseStage2","LungFunct4","LungFunct5","LungFunct6","LungFunct10","LungFunct14","LungFunct15","LungFunct17","LungFunct19","LungFunct20","Disease1","Disease1Treat","Disease2","Disease2Times","Disease3","Disease3Times","Disease4","Disease4Treat","Disease5","Disease5Treat","Disease6","Disease7","Ques1","Ques2","Ques3","Ques5","Smoke1","Smoke2","Smoke3","Smoke4"]]
#
# onehot_df = pd.get_dummies(data, columns=["Factor1","DiseaseHis1","DiseaseHis2","DiseaseHis3","DiseaseHis4","DiseaseHis7","DiseaseStage2","Disease1","Disease1Treat","Disease2","Disease3","Disease4","Disease4Treat","Disease5","Disease5Treat","Disease6","Disease7"])
# onehot_df2 = pd.get_dummies(data_test, columns=["Factor1","DiseaseHis1","DiseaseHis2","DiseaseHis3","DiseaseHis4","DiseaseHis7","DiseaseStage2","Disease1","Disease1Treat","Disease2","Disease3","Disease4","Disease4Treat","Disease5","Disease5Treat","Disease6","Disease7"])


# data = data[["Factor2","Factor4","DiseaseHis2","DiseaseHis2Times","DiseaseHis3","DiseaseHis3Times","DiseaseHis4","DiseaseHis6","DiseaseStage2","LungFunct1","LungFunct2","LungFunct3","LungFunct4","LungFunct5","LungFunct6","LungFunct7","LungFunct8","LungFunct9","LungFunct10","LungFunct11","LungFunct12","LungFunct13","LungFunct14","Disease2Times","Disease3Times","Ques1","Ques2","Ques3","Ques4","Ques5","Smoke1","Smoke2","Smoke3","Smoke4"]]
# data_test = data_test[["Factor2","Factor4","DiseaseHis2","DiseaseHis2Times","DiseaseHis3","DiseaseHis3Times","DiseaseHis4","DiseaseHis6","DiseaseStage2","LungFunct1","LungFunct2","LungFunct3","LungFunct4","LungFunct5","LungFunct6","LungFunct7","LungFunct8","LungFunct9","LungFunct10","LungFunct11","LungFunct12","LungFunct13","LungFunct14","Disease2Times","Disease3Times","Ques1","Ques2","Ques3","Ques4","Ques5","Smoke1","Smoke2","Smoke3","Smoke4"]]
#
# onehot_df = pd.get_dummies(data, columns=["DiseaseHis2","DiseaseHis3","DiseaseHis4","DiseaseStage2"])
# onehot_df2 = pd.get_dummies(data_test, columns=["DiseaseHis2","DiseaseHis3","DiseaseHis4","DiseaseStage2"])


data = data[["Ques2","DiseaseHis3","Factor6","Disease2Times","Smoke3","LungFunct8","Disease3Times","Ques4","LungFunct11","Smoke2","LungFunct7","LungFunct18","Factor3","LungFunct2","DiseaseHis3Times","LungFunct16","LungFunct1","Ques1","LungFunct13","LungFunct17","LungFunct9","LungFunct14","LungFunct12","Factor2","Factor4","LungFunct6","LungFunct20","Smoke4","LungFunct15","LungFunct3"]]
data_test = data_test[["Ques2","DiseaseHis3","Factor6","Disease2Times","Smoke3","LungFunct8","Disease3Times","Ques4","LungFunct11","Smoke2","LungFunct7","LungFunct18","Factor3","LungFunct2","DiseaseHis3Times","LungFunct16","LungFunct1","Ques1","LungFunct13","LungFunct17","LungFunct9","LungFunct14","LungFunct12","Factor2","Factor4","LungFunct6","LungFunct20","Smoke4","LungFunct15","LungFunct3"]]

onehot_df = pd.get_dummies(data, columns=["DiseaseHis3"])
onehot_df2 = pd.get_dummies(data, columns=["DiseaseHis3"])



# onehot_df = pd.get_dummies(data, columns=["Factor1","Factor5","DiseaseHis1","DiseaseHis2","DiseaseHis3","DiseaseHis4","DiseaseHis5","DiseaseHis7","DiseaseStage1","DiseaseStage2","Disease1","Disease1Treat","Disease2","Disease3","Disease4","Disease4Treat","Disease5","Disease5Treat","Disease6","Disease6Treat","Disease7"])
# onehot_df2 = pd.get_dummies(data_test, columns=["Factor1","Factor5","DiseaseHis1","DiseaseHis2","DiseaseHis3","DiseaseHis4","DiseaseHis5","DiseaseHis7","DiseaseStage1","DiseaseStage2","Disease1","Disease1Treat","Disease2","Disease3","Disease4","Disease4Treat","Disease5","Disease5Treat","Disease6","Disease6Treat","Disease7"])

# pd.set_option('display.max_columns', None)

# print
# print onehot_df.info()
# print
# print onehot_df.columns.values
# print
# print onehot_df2.info()
# print
# print onehot_df2.columns.values
# print

# from sklearn.decomposition import PCA
# pca = PCA(n_components=10)
# onehot_df = pca.fit_transform(onehot_df, labels)
# onehot_df2 = pca.transform(onehot_df2)

# from sklearn import feature_selection
# fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=40)      # 40 50 and 70 or 80 gave good results 0.32
# onehot_df = fs.fit_transform(onehot_df, labels)
# onehot_df2 = fs.transform(onehot_df2)

from sklearn.kernel_approximation import RBFSampler
rbf_features = RBFSampler(gamma=0.001, random_state=1)
onehot_df = rbf_features.fit_transform(onehot_df)
onehot_df2 = rbf_features.fit_transform(onehot_df2)

from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss='modified_huber', alpha=0.001, n_iter = 1000)
clf.fit(onehot_df, labels)

op = pd.DataFrame(clf.predict_proba(onehot_df2))
# print op

writer = pd.ExcelWriter('pandas_simple.xlsx', engine='xlsxwriter')
op.to_excel(writer, sheet_name='Sheet1')
