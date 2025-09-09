# -*- coding: utf-8 -*-
# @Time    : 2024/1/3 09:16 AM
# @Author  : Hanwenjie
# @project : code
# @File    : fig_s4_BV.py
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
import warnings
warnings.filterwarnings("ignore")

# Set random seed
seed=2019
df_BV = pd.read_csv(r'../../../data/BV_model.csv',index_col=0)
# Filter data with more than two HI assay repetitions
df_BV = df_BV[df_BV['count']>1].copy()
print(df_BV.shape[0])
print(df_BV['Y'].value_counts())

# Based on Random Forest
# Based on XGBoost
list_feature = ['x_pssmepi1','x_pssmepi2','x_pssmepi3','x_pssmepi4','x_pssmepi5'] + \
               ['x_FAUJ880109', 'x_FAUJ880103', 'x_ZIMJ680104', 'x_ZIMJ680103', 'x_CHOC760101'] + ['x_Nglycosylation','x_rbs']

features = df_BV.loc[:,list_feature]

x_resampled, y_resampled = SMOTE(random_state=seed).fit_resample(features,df_BV['Y'])
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size = 0.2,random_state=seed)
print(y_train.value_counts())


# Train XGBoost model on the training set
clf = LGBMClassifier(**best_params,random_state=seed)
# clf = LGBMClassifier(random_state=seed)
clf.fit(x_train,y_train)

# Save the model
# joblib.dump(clf, "BV_model/BV_xgboost.pkl")
# Use the trained model to make predictions on both training and test sets
train_predict = clf.predict(x_train)
test_predict = clf.predict(x_test)


feature_newname = ['BV-E','BV-A','BV-D','BV-C','BV-B'] + \
                  ['Hydrophobicity', 'Volume', 'Charge', 'Polarity', 'ASA'] + ['N-Gly','RBS']

shap.initjs()
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(x_train)
print(shap_values)
shap_abs0 = np.abs(shap_values[0])
shap_abs1 = np.abs(shap_values[1])
shap_mean0 = np.mean(shap_abs0,axis=0)
shap_mean1 = np.mean(shap_abs1,axis=0)
shap_mean = shap_mean0+shap_mean1
print(shap_mean0)
print(shap_mean1)
print(shap_mean)
shap.summary_plot(shap_values[1], x_train,feature_names=feature_newname, show=False)
# plt.xlim([-3,5])
# Set figure size
plt.gcf().set_size_inches(8, 7)
plt.ylabel("Features")  # Set y-axis label
plt.title("B/Victoria")  # Set title
plt.tight_layout()
plt.savefig(r'./figure/fig_s4_BV.pdf')
plt.show()
