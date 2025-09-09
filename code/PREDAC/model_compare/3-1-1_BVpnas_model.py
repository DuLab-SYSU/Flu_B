# -*- coding: utf-8 -*-
# @Time    : 2025/8/7 4:14 PM
# @Author  : Hanwenjie
# @project : BVcross_Ngly.py
# @File    : 3-1-1_BVpnas_model.py
# @IDE     : PyCharm
# @REMARKS : description text
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
df_BV = pd.read_csv("./data/BV_compare_train.csv",index_col=0)
# Filter data with HI test count greater than 2
df_BV = df_BV[df_BV['count']>1].copy()
print(df_BV.shape[0])
print(df_BV['Y'].value_counts())

# Based on Random Forest
# Based on XGBoost
list_feature = ['x_pssmepi1','x_pssmepi2','x_pssmepi3','x_pssmepi4','x_pssmepi5'] + \
               ['x_FAUJ880109', 'x_FAUJ880103', 'x_ZIMJ680104', 'x_ZIMJ680103', 'x_CHOC760101'] + ['x_Nglycosylation','x_rbs']

features = df_BV.loc[:,list_feature]

x_train, y_train = features,df_BV['Y']
print(y_train.value_counts())

df_BV_test = pd.read_csv("./data/BV_compare_test.csv",index_col=0)
x_test = df_BV_test.loc[:,list_feature]
y_test = df_BV_test.loc[:,'Y']
y_tree = df_BV_test.loc[:,'Y_tree']
y_subs = df_BV_test.loc[:,'Y_subs']
# Train XGBoost model on the training set
clf = LGBMClassifier(**best_params,random_state=seed)
# clf = LGBMClassifier(random_state=seed)
clf.fit(x_train,y_train)

# Save the model
# joblib.dump(clf, "BV_model/BV_xgboost.pkl")
# Use the trained model to make predictions on training and testing sets
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)

# Load model performance file
df_performance = pd.DataFrame()
df_performance['model'] = ['lightgbm', 'tree', 'substitution']
# Evaluate model performance using accuracy (ratio of correctly predicted samples to total samples)
cv = KFold(n_splits=5, shuffle=True, random_state=seed).split(x_train)
index_gbm = df_performance[df_performance['model']=='lightgbm'].index
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
df_performance.to_csv(r"./result/BV_model_performance.csv")
print(confusion_matrix(y_test, test_predict))
