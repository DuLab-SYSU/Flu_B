# -*- coding: utf-8 -*-
# @Time    : 2024/1/3 上午9:19
# @Author  : Hanwenjie
# @project : code
# @File    : fig_s4_BY.py
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

# Specify conversion unit to centimeters
cm = 2.54

feature_newname = ['BY-A','BY-E','BY-B','BY-D','BY-C'] + \
                  ['Hydrophobicity', 'Volume', 'Charge', 'Polarity', 'ASA'] + ['N-Gly','RBS']

shap.initjs()
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(x_train)
shap_abs = np.abs(shap_values)
shap_mean = np.mean(shap_abs,axis=0)
print(shap_values)
print(x_train.shape[0])
print(shap_values.shape)
print(shap_mean)
fig = plt.figure()
shap.summary_plot(shap_values, x_train, feature_names=feature_newname, show=False)
# plt.xlim([-1.5,1.5])
# Set figure size
plt.gcf().set_size_inches(8, 7)
# plt.rcParams['font.family'] = 'Arial'  # Set font to Arial (or other if desired)
# plt.xlabel("SHAP value (impact on model output)", fontsize=7)  # Set x-axis label
plt.ylabel("Features")  # Set y-axis label
plt.title("B/Yamagata")  # Set title

plt.tight_layout()
plt.savefig(r'./figure/fig_s4_BY.pdf')
plt.show()
