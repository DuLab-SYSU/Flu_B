# -*- coding: utf-8 -*-
# @Time    : 2024/1/3 上午9:19
# @Author  : Hanwenjie
# @project : code
# @File    : BY_xgb_new.py
# @IDE     : PyCharm

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

# set random seed
seed=2019
df_BY = pd.read_csv(r'../data/PREDAC/BY_model.csv',index_col=0)
#seletc pair with more than 2 results of HI assay
df_BY = df_BY[df_BY['count']>1].copy()
print(df_BY.shape[0])
print(df_BY['Y'].value_counts())
#features
list_feature = ['x_pssmepi1','x_pssmepi2','x_pssmepi3','x_pssmepi4','x_pssmepi5'] + \
               ['x_FAUJ880109', 'x_PONJ960101', 'x_ZIMJ680104', 'x_CHAM820101', 'x_CHOC760102'] + ['x_Nglycosylation','x_rbs']

features = df_BY.loc[:,list_feature]

# balance positive and negative samples by SMOTE
x_resampled, y_resampled = SMOTE(random_state=seed).fit_resample(features,df_BY['Y'])
# split training and test dataset
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size = 0.2,random_state=seed)

#tune params

# best params tuned by above steps

# train XGBoost model on training dataset
clf = XGBClassifier(**best_params,random_state=seed)
clf.fit(x_train,y_train)


# predict on training and test dataset
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)

# model performance
df_performance = pd.DataFrame()
cv = KFold(n_splits=5, shuffle=True, random_state=seed).split(x_train)
# accuracy
index_gbm = df_performance[df_performance['model']=='xgboost'].index
df_performance.loc[index_gbm,'accuracy_test'] = accuracy_score(y_test,test_predict)
df_performance.loc[index_gbm,'accuracy_cv'] = cross_val_score(clf, x_train, y_train, cv=cv,scoring='accuracy').mean()

# precision
df_performance.loc[index_gbm,'precision_test'] = metrics.precision_score(y_test,test_predict)
df_performance.loc[index_gbm,'precision_cv'] = cross_val_score(clf, x_train, y_train, cv=cv, n_jobs=-1,scoring='precision').mean()

#recall
df_performance.loc[index_gbm,'recall_test'] = metrics.recall_score(y_test,test_predict)
df_performance.loc[index_gbm,'recall_cv'] = cross_val_score(clf, x_train, y_train, cv=cv, n_jobs=-1,scoring='recall').mean()

#auc
df_performance.loc[index_gbm,'auc_test'] = metrics.roc_auc_score(y_test,test_predict)
df_performance.loc[index_gbm,'auc_cv'] = cross_val_score(clf, x_train, y_train, cv=cv,scoring='roc_auc').mean()

#f1-score
df_performance.loc[index_gbm,'f1-score_test'] = metrics.f1_score(y_test,test_predict)
df_performance.loc[index_gbm,'f1-score_cv'] = cross_val_score(clf, x_train, y_train, cv=cv, n_jobs=-1,scoring='f1').mean()
# export model performance dataset
df_performance.to_csv(r"../result/PREDAC/BY_model_performance.csv")
