# -*- coding: utf-8 -*-
# @Time    : 2025/8/5 下午5:18
# @Author  : Hanwenjie
# @project : BVcross_Ngly.py
# @File    : 3-1-1_BYelifecompare_seed62.py
# @IDE     : PyCharm
# @REMARKS : description text
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
# import PREDAC predicted data
df_predac = pd.read_csv(r"./data/"
                        r"Yam_test_predac.tsv", sep='\t')
# import BMDS predicted data
df_bmds = pd.read_csv(r"./data/"
                      r"Yam_test_elife.csv", index_col=0)
df_bmds = df_bmds[['virus_1', 'virus_2', 'Y_bmds']].copy()

# merge PREDAC and BMDS predicted data
df = pd.merge(df_predac, df_bmds, on=['virus_1', 'virus_2'], how='inner')
df.reset_index(inplace=True, drop=True)

# extract true and predicted antigenic relationship label
y_predac = df['Y_predac'].copy()
y_bmds = df['Y_bmds'].copy()
y_true = df['Y_elife'].copy()

# performance file
df_performance = pd.DataFrame()
df_performance['model'] = ['PREDAC', 'BMDS']
# calculate performance indicator
# accuracy
df_performance.loc[0,'accuracy_test'] = accuracy_score(y_true,y_predac)
df_performance.loc[1,'accuracy_test'] = accuracy_score(y_true,y_bmds)
# precision
df_performance.loc[0,'precision_test'] = precision_score(y_true,y_predac)
df_performance.loc[1,'precision_test'] = precision_score(y_true,y_bmds)
# recall
df_performance.loc[0,'recall_test'] = recall_score(y_true,y_predac)
df_performance.loc[1,'recall_test'] = recall_score(y_true,y_bmds)
# auc
df_performance.loc[0,'auc_test'] = roc_auc_score(y_true,y_predac)
df_performance.loc[1,'auc_test'] = roc_auc_score(y_true,y_bmds)
# f1-score
df_performance.loc[0,'f1-score_test'] = f1_score(y_true,y_predac)
df_performance.loc[1,'f1-score_test'] = f1_score(y_true,y_bmds)
# confusion_matrix
tn_predac, fp_predace, fn_predac, tp_predace = confusion_matrix(y_true,y_predac).ravel()
tn_bmds, fp_bmds, fn_bmds, tp_bmds = confusion_matrix(y_true,y_bmds).ravel()
df_performance.loc[0,'num of correction'] = tn_predac + tp_predace
df_performance.loc[1,'num of correction'] = tn_bmds + tp_bmds

df_performance.to_csv(r"./result/"
                      r"Yam_compare.csv", index=False)
match_rate = df.query('Y_predac == Y_bmds').copy().shape[0]/df.shape[0]
print("match rate:", match_rate)