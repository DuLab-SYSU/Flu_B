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
other_params = {'learning_rate':np.arange(0.01,0.2,0.01),
                'n_estimators':list(range(10,1000,10)),
                'max_depth':list(range(1,15,1)),
                'min_child_weight':list(range(1,15,1)),
                'gamma':np.arange(0,1,0.1),
                'subsample':np.arange(0.5,1,0.1),
                'colsample_bytree':np.arange(0.5,1,0.1),
                # 'objective':'binary:logistic',
                'reg_alpha':[0,1e-5, 1e-2, 0.1, 1,2,3,100],
                'reg_lambda':[1e-5, 1e-2, 0.1, 1,2,3,100],
                # 'scale_pos_weight':1,
                # 'n_jobs':23
                }


start = time.time()
cv = KFold(n_splits=5,shuffle=True, random_state=seed).split(x_train)
model = XGBClassifier(random_state=seed)
# optimized_xgb = GridSearchCV(estimator=model, param_grid =param_test1, scoring='roc_auc',n_jobs=23, cv=cv)
optimized_xgb = RandomizedSearchCV(estimator=model, param_distributions=other_params, n_iter=2000,scoring='roc_auc',n_jobs=30, cv=cv,random_state=seed)
optimized_xgb.fit(x_train,y_train)
# print(optimized_xgb.grid_scores_)
print(optimized_xgb.best_params_)
print(optimized_xgb.best_score_)
end = time.time()
print("The runing time of this process is {}min".format((end-start)/60))

# best params tuned by above steps
best_params = {'learning_rate':0.12,
                'n_estimators':450,
                'max_depth':6,
                'min_child_weight':2,
                'gamma':0.0,
                'subsample':0.6,
                'colsample_bytree':0.5,
                # 'objective':'binary:logistic',
                'reg_alpha':3,
                'reg_lambda':3,
                # 'scale_pos_weight':1,
                # 'n_jobs':23
                }


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
