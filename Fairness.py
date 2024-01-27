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
from aif360.metrics import ClassificationMetric
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame(features, columns = ACSEmployment.features)
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

# print(data_for_aif.features,'000', data_for_aif.features.shape)
# print(data_for_aif.labels,'000', data_for_aif.labels.shape)
# train_data, test_data, labels_train, labels_test = train_test_split(data_for_aif.features,
#                                                                       data_for_aif.labels,                   
#                                                                       test_size=0.3,
#                                                                       random_state=42)

data_len = data.shape[0]
print(data_len,'111')
train, test= data_for_aif.split([0.7], shuffle=False)
print(train.features.shape,'222', test.features.shape)
acc_list = []
for C in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
  acc_avglist = []
  for i in range(5):
       train_train, train_val = train.split([0.8], shuffle=True, seed=i)

       logit_model = LogisticRegression(C=C, max_iter=1000)
       logit_model.fit(train_train.features, train_train.labels.ravel())
       predict_logit = logit_model.predict(train_val.features)
       probs_logit = pd.DataFrame(logit_model.predict_proba(train_val.features))
       report = classification_report(train_val.labels, predict_logit, output_dict=True)
       acc=report['accuracy']
       acc_avglist.append(acc)
  acc_in = np.mean(acc_avglist)
  print(acc_in)
  acc_list.append(acc_in)
plt.plot([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], acc_list)
plt.xlabel('C value')
plt.ylabel('Accuracy')
plt.title('Logistic Regression Model Accuracy with different C values')
plt.show()
    
print(np.mean(acc_list), np.std(acc_list))

    