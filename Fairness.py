import folktables
from folktables import ACSDataSource
import numpy as np

'''
 (Age) must be greater than 16 and less than 90, and (Person weight) must be
 greater than or equal to 1
'''
 
def employment_filter(data):
   """
    Filters for the employment prediction task
   """
   df = data
   df = df[df['AGEP'] > 16]
   df = df[df['AGEP'] < 90]
   df = df[df['PWGTP'] >= 1]
   return df

ACSEmployment = folktables.BasicProblem(
 features=[
   'AGEP', #age; for range of values of features please check Appendix B.4 of Retiring Adult: New Datasets for Fair Machine Learning NeurIPS 2021 paper
   'SCHL', #educational attainment
   'MAR' , #marital status
   'RELP', #relationship
   'DIS' , #disability recode
   'ESP' , #employment status of parents
   'CIT' , #citizenship status
   'MIG' , #mobility status (lived here 1 year ago)
   'MIL' , #military service
   'ANC' , #ancestry recode
   'NATIVITY', #nativity
   'DEAR', #hearing difficulty
   'DEYE', #vision difficulty
   'DREM', #cognitive difficulty
   'SEX' , #sex
   'RAC1P', #recoded detailed race code
   'GCL' , #grandparents living with grandchildren
    ],
 target='ESR', #employment status recode
 target_transform=lambda x: x == 1,
 group='DIS',
 preprocess=employment_filter,
 postprocess=lambda x: np.nan_to_num(x,-1),
 )
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(states=["FL"], download=True) #data for Florida state
features, label, group = ACSEmployment.df_to_numpy(acs_data)
# print(features.shape, label.shape, group.shape, len(ACSEmployment.features))

from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric, DatasetMetric
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame(features, columns = ACSEmployment.features)
print(np.unique(data['RAC1P']),'---111---') 
data['label'] = label
print('original data', data.shape)
print(data)
favorable_classes = [True]
protected_attribute_names = [ACSEmployment.group]
privileged_classes = np.array([[1]])
data_for_aif = StandardDataset(data, 'label', favorable_classes = favorable_classes,
protected_attribute_names = protected_attribute_names,
privileged_classes = privileged_classes)
print('data aif\n',data_for_aif,'---')

print('data aif features',data_for_aif.features.shape)
print('data aif labels',data_for_aif.labels.shape)
privileged_groups = [{'DIS': 1}]
unprivileged_groups = [{'DIS': 2}]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import sklearn
import numpy as np


data_len = data.shape[0]
print(data_len,'111')
train, test= data_for_aif.split([0.7], shuffle=False)

#========================================task1=================================================
# acc_list = []
# fairness_list = []
# x = [0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
# for C in x:
#   acc_avglist = []
#   fairness_avglist = []
#   for i in range(5):
#        train_train, train_val = train.split([0.8], shuffle=True, seed=i)

#        logit_model = LogisticRegression(C=C)
#        logit_model.fit(train_train.features, train_train.labels.ravel())
#        predict_logit = logit_model.predict(train_val.features)
#        probs_logit = pd.DataFrame(logit_model.predict_proba(train_val.features))
#        report = classification_report(train_val.labels, predict_logit, output_dict=True)
#        val_pred = train_val.copy()
#        val_pred.labels = predict_logit
#        metric = ClassificationMetric(train_val, val_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
#        fairness_avglist.append(metric.true_positive_rate_difference())
#        acc=report['accuracy']
#        acc_avglist.append(acc)
#   acc_in = np.mean(acc_avglist)
#   print(acc_in, fairness_avglist)
#   acc_list.append(acc_in)
#   fairness_avglist = np.mean(fairness_avglist)
#   fairness_list.append(fairness_avglist)
# print('-=-',x, acc_list)
# for i in range(len(acc_list)):
#   print(x[i], '->', acc_list[i], fairness_list[i])
# plt.plot(x, acc_list, marker='p', label='Accuracy')
# plt.plot(x, fairness_list, marker='*', label='Fairness')
# plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# plt.xscale('log')
# plt.xlabel('C value')
# plt.ylabel('Accuracy and Fairness')
# plt.title('Logistic Regression Model with different C values')
# plt.legend()
# plt.savefig('Logistic Regression Model with different C values.png')
# plt.show()

# logit_model = LogisticRegression(C=1000)
# logit_model.fit(train.features, train.labels.ravel())
# predict_logit = logit_model.predict(test.features)
# probs_logit = pd.DataFrame(logit_model.predict_proba(test.features))
# report = classification_report(test.labels, predict_logit, output_dict=True)
# acc=report['accuracy']
# test_pred = test.copy()
# test_pred.labels = predict_logit
# metric = ClassificationMetric(test, test_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
# print('accuracy', acc)
# print('fairness', metric.true_positive_rate_difference())

# logit_model = LogisticRegression(C=1e-9)
# logit_model.fit(train.features, train.labels.ravel())
# predict_logit = logit_model.predict(test.features)
# probs_logit = pd.DataFrame(logit_model.predict_proba(test.features))
# report = classification_report(test.labels, predict_logit, output_dict=True)
# acc=report['accuracy']
# test_pred = test.copy()
# test_pred.labels = predict_logit
# metric = ClassificationMetric(test, test_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
# print('accuracy', acc)
# print('fairness', metric.true_positive_rate_difference())

#======================================task2===================================================

from aif360.algorithms.preprocessing.reweighing import Reweighing
RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
# train_transformed = RW.fit_transform(train)
# print("subgroup weights", np.unique(train_transformed.instance_weights))
# acc_list = []
# fairness_list = []
# x = [0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
# for C in x:
#   acc_avglist = []
#   fairness_avglist = []
#   for i in range(5):
#        train_train, train_val = train.split([0.8], shuffle=True, seed=i)
#        logit_model = LogisticRegression(C=C)
#        train_train = RW.fit_transform(train_train)
#        logit_model.fit(train_train.features, train_train.labels.ravel(),sample_weight=train_train.instance_weights)
#        predict_logit = logit_model.predict(train_val.features)
#        probs_logit = pd.DataFrame(logit_model.predict_proba(train_val.features))
#        report = classification_report(train_val.labels, predict_logit, output_dict=True)
#        val_pred = train_val.copy()
#        val_pred.labels = predict_logit
#        metric = ClassificationMetric(train_val, val_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
#        fairness_avglist.append(metric.true_positive_rate_difference())
#        acc=report['accuracy']
#        acc_avglist.append(acc)
#   acc_in = np.mean(acc_avglist)
#   acc_list.append(acc_in)
#   fairness_in = np.mean(fairness_avglist)
#   print(acc_in, fairness_in)
#   fairness_list.append(fairness_in)
# print('-=-',x, acc_list)
# for i in range(len(acc_list)):
#   print(x[i], '->', acc_list[i], fairness_list[i])
# plt.plot(x, acc_list, marker='p', label='Accuracy')
# plt.plot(x, fairness_list, marker='*', label='Fairness')
# plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# plt.xscale('log')
# plt.xlabel('C value')
# plt.ylabel('Accuracy and Fairness')
# plt.title('Fairness-awared Logistic Regression Model with different C values')
# plt.legend()
# plt.savefig('Task2 Logistic Regression Model with different C values.png')
# plt.show()

# logit_model = LogisticRegression(C=0.1)
# train = RW.fit_transform(train)
# logit_model.fit(train.features, train.labels.ravel(), sample_weight=train.instance_weights)
# predict_logit = logit_model.predict(test.features)
# probs_logit = pd.DataFrame(logit_model.predict_proba(test.features))
# report = classification_report(test.labels, predict_logit, output_dict=True)
# acc=report['accuracy']
# test_pred = test.copy()
# test_pred.labels = predict_logit
# metric = ClassificationMetric(test, test_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
# print('accuracy', acc)
# print('fairness', metric.true_positive_rate_difference())



#======================================task3====================================================

#=======================================beyond==================================================

unprivileged_groups = [{'RAC1P': 1}]
privileged_groups = [{'RAC1P': 9}]

logit_model = LogisticRegression(C=0.1)
train = RW.fit_transform(train)
logit_model.fit(train.features, train.labels.ravel(), sample_weight=train.instance_weights)
predict_logit = logit_model.predict(test.features)
probs_logit = pd.DataFrame(logit_model.predict_proba(test.features))
report = classification_report(test.labels, predict_logit, output_dict=True)
acc=report['accuracy']
test_pred = test.copy()
test_pred.labels = predict_logit
metric = ClassificationMetric(test, test_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
print('accuracy', acc)
print('fairness', metric.true_positive_rate_difference())

