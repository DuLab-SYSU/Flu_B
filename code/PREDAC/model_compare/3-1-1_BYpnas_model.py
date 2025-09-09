# -*- coding: utf-8 -*-
# @Time    : 2025/8/7 4:19 PM
# @Author  : Hanwenjie
# @project : BVcross_Ngly.py
# @File    : 3-1-1_BYpnas_model.py
# @IDE     : PyCharm
# @REMARKS : Notes
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
# import shap
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
import warnings
warnings.filterwarnings("ignore")

# Set random seed
seed=2019
df_BY = pd.read_csv("./data/BY_compare_train.csv",index_col=0)
# Filter data with more than two HI experiments
df_BY = df_BY[df_BY['count']>1].copy()
print(df_BY.shape[0])
print(df_BY['Y'].value_counts())
# Based on Random Forest
list_feature = ['x_pssmepi1','x_pssmepi2','x_pssmepi3','x_pssmepi4','x_pssmepi5'] + \
               ['x_FAUJ880109', 'x_PONJ960101', 'x_ZIMJ680104', 'x_CHAM820101', 'x_CHOC760102'] + ['x_Nglycosylation','x_rbs']

features = df_BY.loc[:,list_feature]

# Balance positive and negative samples using SMOTE
x_train, y_train = features,df_BY['Y']


df_BY_test = pd.read_csv("./data/BY_compare_test.csv",index_col=0)
x_test = df_BY_test.loc[:,list_feature]
y_test = df_BY_test.loc[:,'Y']
y_tree = df_BY_test.loc[:,'Y_tree']
y_subs = df_BY_test.loc[:,'Y_subs']
# Train XGBoost model on the training set
clf = XGBClassifier(**best_params,random_state=seed)
# clf = XGBClassifier(random_state=seed)
clf.fit(x_train,y_train)

# Save the model
# joblib.dump(clf, "BY_model/BY_xgboost.pkl")
# Use the trained model to predict on the training and testing sets
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)

# Read model performance file
df_performance = pd.DataFrame()
df_performance['model'] = ['lightgbm', 'tree', 'substitution']
# Evaluate model performance using accuracy (the proportion of correctly predicted samples among all samples)
cv = KFold(n_splits=5, shuffle=True, random_state=seed).split(x_train)
print('Cross-validation accuracy:',cross_val_score(clf, x_train, y_train, cv=cv,scoring='accuracy').mean())
df_performance.loc[0,'accuracy_test'] = accuracy_score(y_test,test_predict)
df_performance.loc[1,'accuracy_test'] = accuracy_score(y_test,y_tree)
df_performance.loc[2,'accuracy_test'] = accuracy_score(y_test,y_subs)
# cv = KFold(n_splits=5, shuffle=True, random_state=seed).split(x_train)
# df_performance.loc[0,'accuracy_cv'] = cross_val_score(clf, x_train, y_train, cv=cv,scoring='accuracy').mean()

cv = KFold(n_splits=5,shuffle=True, random_state=seed).split(x_train)
df_performance.loc[0,'precision_test'] = metrics.precision_score(y_test,test_predict)
df_performance.loc[1,'precision_test'] = metrics.precision_score(y_test,y_tree)
df_performance.loc[2,'precision_test'] = metrics.precision_score(y_test,y_subs)
# cv = KFold(n_splits=5, shuffle=True, random_state=seed).split(x_train)
# df_performance.loc[0,'precision_cv'] = cross_val_score(clf, x_train, y_train, cv=cv, n_jobs=-1,scoring='precision').mean()

cv = KFold(n_splits=5,shuffle=True, random_state=seed).split(x_train)
df_performance.loc[0,'recall_test'] = metrics.recall_score(y_test,test_predict)
df_performance.loc[1,'recall_test'] = metrics.recall_score(y_test,y_tree)
df_performance.loc[2,'recall_test'] = metrics.recall_score(y_test,y_subs)
# cv = KFold(n_splits=5, shuffle=True, random_state=seed).split(x_train)
# df_performance.loc[0,'recall_cv'] = cross_val_score(clf, x_train, y_train, cv=cv, n_jobs=-1,scoring='recall').mean()

cv = KFold(n_splits=5, shuffle=True, random_state=seed).split(x_train)
print('Cross-validation roc_auc:',cross_val_score(clf, x_train, y_train, cv=cv,scoring='roc_auc').mean())
df_performance.loc[0,'auc_test'] = metrics.roc_auc_score(y_test,test_predict)
df_performance.loc[1,'auc_test'] = metrics.roc_auc_score(y_test,y_tree)
df_performance.loc[2,'auc_test'] = metrics.roc_auc_score(y_test,y_subs)
# cv = KFold(n_splits=5, shuffle=True, random_state=seed).split(x_train)
# df_performance.loc[0,'auc_cv'] = cross_val_score(clf, x_train, y_train, cv=cv,scoring='roc_auc').mean()

cv = KFold(n_splits=5,shuffle=True, random_state=seed).split(x_train)
df_performance.loc[0,'f1-score_test'] = metrics.f1_score(y_test,test_predict)
df_performance.loc[1,'f1-score_test'] = metrics.f1_score(y_test,y_tree)
df_performance.loc[2,'f1-score_test'] = metrics.f1_score(y_test,y_subs)
# cv = KFold(n_splits=5, shuffle=True, random_state=seed).split(x_train)
# df_performance.loc[0,'f1-score_cv'] = cross_val_score(clf, x_train, y_train, cv=cv, n_jobs=-1,scoring='f1').mean()

tn, fp, fn, tp = confusion_matrix(y_test, test_predict).ravel()
tn_tree, fp_tree, fn_tree, tp_tree = confusion_matrix(y_test, y_tree).ravel()
tn_subs, fp_subs, fn_subs, tp_subs = confusion_matrix(y_test,y_subs).ravel()
df_performance.loc[0,'num of correction'] = tn + tp
df_performance.loc[1,'num of correction'] = tn_tree + tp_tree
df_performance.loc[2,'num of correction'] = tn_subs + tp_subs
df_performance.to_csv(r"./result/BY_model_performance.csv")
print(confusion_matrix(y_test, test_predict))
