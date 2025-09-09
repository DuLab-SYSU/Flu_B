# -*- coding: utf-8 -*-
# @Time    : 2024/1/3 上午9:19
# @Author  : Hanwenjie
# @project : code
# @File    : BY_xgb_new.py
# @IDE     : PyCharm
# @REMARKS : Notes
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, \
     confusion_matrix

# Set random seed
seed=2019
df_BY = pd.read_csv(r'../../../data/BY_model.csv',index_col=0)
# Filter data with more than two HI experiments
df_BY = df_BY[df_BY['count']>1].copy()
print(df_BY.shape[0])
print(df_BY['Y'].value_counts())
# Based on Random Forest
list_feature = ['x_pssmepi1','x_pssmepi2','x_pssmepi3','x_pssmepi4','x_pssmepi5'] + \
               ['x_FAUJ880109', 'x_PONJ960101', 'x_ZIMJ680104', 'x_CHAM820101', 'x_CHOC760102'] + ['x_Nglycosylation','x_rbs']

features = df_BY.loc[:,list_feature]


# Use SMOTE to balance positive and negative samples
x_resampled, y_resampled = SMOTE(random_state=seed).fit_resample(features,df_BY['Y'])
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size = 0.2,random_state=seed)

# Train XGBoost model on training set
clf = XGBClassifier(**best_params,random_state=seed)
# clf = XGBClassifier(random_state=seed)
clf.fit(x_train,y_train)


# Save the model
# joblib.dump(clf, "BY_model/BY_xgboost.pkl")
# Use the trained model to predict on both training and test sets
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)


# Load model performance file
df_performance = pd.read_csv(r"../../../data/BY_model_performance.csv",index_col=0)
print(df_performance.columns)
print(df_performance)

# Evaluate model performance using accuracy (proportion of correctly predicted samples among all predictions)
print('XGBoost classification results:')
cv = KFold(n_splits=5, shuffle=True, random_state=seed).split(x_train)
print('Training accuracy:',accuracy_score(y_train,train_predict))
print('Test accuracy:',accuracy_score(y_test,test_predict))
index_xgb = df_performance[df_performance['model']=='xgboost'].index
df_performance.loc[index_xgb,'accuracy_test'] = accuracy_score(y_test,test_predict)
print('Cross-validation accuracy:',cross_val_score(clf, x_train, y_train, cv=cv,scoring='accuracy').mean())
cv = KFold(n_splits=5, shuffle=True, random_state=seed).split(x_train)
df_performance.loc[index_xgb,'accuracy_cv'] = cross_val_score(clf, x_train, y_train, cv=cv,scoring='accuracy').mean()

cv = KFold(n_splits=5,shuffle=True, random_state=seed).split(x_train)
print('Training precision:',metrics.precision_score(y_train,train_predict))
print('Test precision:',metrics.precision_score(y_test,test_predict))
df_performance.loc[index_xgb,'precision_test'] = metrics.precision_score(y_test,test_predict)
print('Cross-validation precision:',cross_val_score(clf, x_train, y_train, cv=cv, n_jobs=-1,scoring='precision').mean())
cv = KFold(n_splits=5, shuffle=True, random_state=seed).split(x_train)
df_performance.loc[index_xgb,'precision_cv'] = cross_val_score(clf, x_train, y_train, cv=cv, n_jobs=-1,scoring='precision').mean()

cv = KFold(n_splits=5,shuffle=True, random_state=seed).split(x_train)
print('Training recall:',metrics.recall_score(y_train,train_predict))
print('Test recall:',metrics.recall_score(y_test,test_predict))
df_performance.loc[index_xgb,'recall_test'] = metrics.recall_score(y_test,test_predict)
print('Cross-validation recall:',cross_val_score(clf, x_train, y_train, cv=cv, n_jobs=-1,scoring='recall').mean())
cv = KFold(n_splits=5, shuffle=True, random_state=seed).split(x_train)
df_performance.loc[index_xgb,'recall_cv'] = cross_val_score(clf, x_train, y_train, cv=cv, n_jobs=-1,scoring='recall').mean()

cv = KFold(n_splits=5, shuffle=True, random_state=seed).split(x_train)
print('Test roc_auc:',metrics.roc_auc_score(y_test,test_predict))
df_performance.loc[index_xgb,'auc_test'] = metrics.roc_auc_score(y_test,test_predict)
print('Cross-validation roc_auc:',cross_val_score(clf, x_train, y_train, cv=cv,scoring='roc_auc').mean())
cv = KFold(n_splits=5, shuffle=True, random_state=seed).split(x_train)
df_performance.loc[index_xgb,'auc_cv'] = cross_val_score(clf, x_train, y_train, cv=cv,scoring='roc_auc').mean()

cv = KFold(n_splits=5,shuffle=True, random_state=seed).split(x_train)
print('Training f1:',metrics.f1_score(y_train,train_predict))
print('Test f1:',metrics.f1_score(y_test,test_predict))
df_performance.loc[index_xgb,'f1-score_test'] = metrics.f1_score(y_test,test_predict)
print('Cross-validation f1:',cross_val_score(clf, x_train, y_train, cv=cv, n_jobs=-1,scoring='f1').mean())
cv = KFold(n_splits=5, shuffle=True, random_state=seed).split(x_train)
df_performance.loc[index_xgb,'f1-score_cv'] = cross_val_score(clf, x_train, y_train, cv=cv, n_jobs=-1,scoring='f1').mean()
# df_performance.to_csv(r"./result/BY_model_performance.csv")


# Predict antigenic relationships of all strain pairs
# Import proportional sampling data
data = pd.read_csv(r'../../../data/BYHA1_predictdata.csv',index_col=0)
print(list_feature)
df_predict = data.loc[:,list_feature].copy()
data_predict = clf.predict(df_predict)
data_predict_prob = clf.predict_proba(df_predict)
data['prob_similar'] = data_predict_prob[:,0]
data['prob_dissimilarity'] = data_predict_prob[:,1]
data['anti_ratio'] = data['prob_similar']/data['prob_dissimilarity']
data['log_ratio'] = np.log(data['anti_ratio'])
data['similarity'] = data_predict
print(data['similarity'].value_counts())

# Export proportional sampling data
data.to_csv(r'./result/BYHA1_predictdata.csv')


# Prepare data for MCL
# Import count sampling data
df = pd.read_csv(r'../../../data/BY_predictdata_1988_dropoutlier.csv',index_col=0)
print(df.shape[0])
# df['log10_ratio'] = np.log10(df['anti_ratio'])
df_similar = df[df['anti_ratio'] > 1].copy()
df_similar.reset_index(inplace=True,drop=True)
print(df_similar.shape[0])
print(df_similar.head())
print(df['similarity'].value_counts())
df_MCL = df_similar.loc[:,['virus1_description','virus2_description','anti_ratio']]
print(df_MCL.shape[0])
# print(df_MCL.head())

# Export count sampling data
df_MCL.to_csv(r'./result/BY_predictdata_1988_dropoutlier_MCL.csv')