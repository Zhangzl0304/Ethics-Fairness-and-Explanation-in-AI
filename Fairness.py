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
from sklearn import preprocessing

data = pd.DataFrame(features, columns = ACSEmployment.features)
data['label'] = label
# print('original data', data.shape)
# print(data)
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



#======================================task3====================================================

# model_acc = [0.515544550228408, 0.515544550228408, 0.6161892157281305, 0.6986807838449388, 0.7006958971950648, 
#              0.7256628100584894, 0.7414592494556633, 0.7512188874183494,  0.7539768603509371, 0.7546770268539469,
#              0.754181787132306, 0.7545745634632627, 0.7543610980660034, 0.7548221833240832, 0.7540793237416216,
#              0.7543696366818938]
# model_fairness = [0,0,0.04231127614666939, 0.20775891202815422, 0.2201733284977334, 0.2385748160573387,0.3213902316100842,
#                   0.5025881386230556, 0.5857838864708226, 0.6043085125181762, 0.6034620205888164, 0.60137861140189,
#                   0.6066480500357869, 0.6032614222227031, 0.6051090391579286, 0.6107928913388355]
# x = [0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]

# fmodel_acc = [0.515544550228408, 0.515544550228408, 0.5804807240746275, 0.7009947487512275,0.7031550185714897,
#               0.7259872774623234, 0.7331597148102292, 0.7280877769713529, 0.7222217478546729, 0.7209238782393375,
#               0.7205908722196133, 0.7202151731204373, 0.720420099901806, 0.7203261751270119, 0.7203859454382445,
#               0.7202493275839985]
# fmodel_fairness = [0,0,0.024365137700911022, 0.2047737865828272, 0.21322681199875176, 0.20094723303178502,0.13265742610008646,
#                    0.06355517363622021, 0.009564662792059387, -0.0044976697010731485, -0.00616764026071126,-0.004765514313744034,
#                    -0.0051453176783304855, -0.008302719175295592, -0.008026485952509588, -0.007388363920346808]

# fciterion = []
# for i in range(len(x)):
#   fciterion.append(fmodel_acc[i]-abs(fmodel_fairness[i]))
# citerion = []
# for i in range(len(x)):
#   citerion.append(model_acc[i]-abs(model_fairness[i]))

# print(fciterion,'ff')
# print(citerion,'cc')
# plt.plot(x, citerion, marker='^', label='Standard Model')
# plt.plot(x, fciterion, marker='o', label='Fairness-aware Model')
# plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# plt.xscale('log')
# plt.xlabel('C value')
# plt.ylabel('Model Select Criterion')
# plt.title('Proposed criterion with different C values')
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
# print('cri',acc-abs(metric.true_positive_rate_difference()))

#=======================================beyond binary==================================================
# all_groups = [{'RAC1P': 1.0},{'RAC1P': 2.0},{'RAC1P': 3.0},{'RAC1P': 4.0},{'RAC1P': 5.0},{'RAC1P': 6.0},{'RAC1P': 7.0},{'RAC1P': 8.0},{'RAC1P': 9.0}]
# # unprivileged_groups = [{'RAC1P': 1.0},{'RAC1P': 2.0},{'RAC1P': 3.0},{'RAC1P': 4.0},{'RAC1P': 5.0},{'RAC1P': 6.0},{'RAC1P': 7.0},{'RAC1P': 8.0}]
# # privileged_groups = [{'RAC1P': 9.0}]
# acc_list = []
# fairness_list = []
# x = [0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
# for C in x:
#   acc_avglist = []
#   fairness_avglist = []
#   for i in range(5):
#       train_train, train_val = train.split([0.8], shuffle=True, seed=i)
#       logit_model = LogisticRegression(C=C)
#       logit_model.fit(train_train.features, train_train.labels.ravel(),sample_weight=train_train.instance_weights)
#       predict_logit = logit_model.predict(train_val.features)
#       probs_logit = pd.DataFrame(logit_model.predict_proba(train_val.features))
#       report = classification_report(train_val.labels, predict_logit, output_dict=True)
#       val_pred = train_val.copy()
#       val_pred.labels = predict_logit
#       fairness_in_diff = []
#       fairness_eachrace = {}
#       for i in range(9):
#         unprivileged_groups = [all_groups[i]]
#         print(unprivileged_groups)
#         all_groups1 = all_groups.copy()
#         all_groups1.remove(unprivileged_groups[0])
#         print(all_groups1)
#         metric = ClassificationMetric(train_val, val_pred, unprivileged_groups=unprivileged_groups, privileged_groups=all_groups1)
#         fairness_in_diff.append(metric.true_positive_rate_difference())
#         fairness_eachrace[unprivileged_groups] = metric.true_positive_rate_difference()
#       print(fairness_in_diff)
#       fairness_in_diff1 = [0 if np.isnan(x) else x for x in fairness_in_diff]
#       fairness_in_diff1 = [abs(x) for x in fairness_in_diff1]
#       print(fairness_in_diff1,'diff')
#       fairness_avg_diff = np.mean(fairness_in_diff1)
#       fairness_avglist.append(fairness_avg_diff)
#       acc=report['accuracy']
#       acc_avglist.append(acc)
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
# plt.title('Standard Logistic Regression Model with different C values on Multiclass')
# plt.legend()
# plt.savefig('Beyond.png')
# plt.show()

# privileged_groups = [{'RAC1P': 4.0}]
# unprivileged_groups = [{'RAC1P': 1.0}, {'RAC1P': 2.0}, {'RAC1P': 3.0}, {'RAC1P': 5.0}, {'RAC1P': 6.0}, {'RAC1P': 7.0}, {'RAC1P': 8.0}, {'RAC1P': 9.0}]
# logit_model = LogisticRegression(C=100000)
# # train = RW.fit_transform(train)
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

#=======================================beyond Florida==================================================
# data_for_aif = RW.fit_transform(data_for_aif)
# logit_model = LogisticRegression(C=1e-8)
# logit_model.fit(data_for_aif.features, data_for_aif.labels.ravel())

# TX_data = data_source.get_data(states=["TX"], download=True) #data for Tex state
# features1, label1, group1 = ACSEmployment.df_to_numpy(TX_data)
# data_TX = pd.DataFrame(features1, columns = ACSEmployment.features)
# data_TX['label'] = label1
# privileged_classes = np.array([[1]])
# data_for_TX = StandardDataset(data_TX, 'label', favorable_classes = favorable_classes,
# protected_attribute_names = protected_attribute_names,
# privileged_classes = privileged_classes)
# print(data_for_TX.features.shape,'txtxtx')
# print(data_for_aif.features.shape,'txtxtx')
# predict_logit = logit_model.predict(data_for_TX.features)
# report = classification_report(data_for_TX.labels, predict_logit, output_dict=True)
# test_pred = data_for_TX.copy()
# test_pred.labels = predict_logit
# metric = ClassificationMetric(data_for_TX, test_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
# print('acc',report['accuracy'])
# print('fairness', metric.true_positive_rate_difference())
