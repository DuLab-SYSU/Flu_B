# -*- coding: utf-8 -*-
# @Time    : 2024/1/8 下午2:33
# @Author  : Hanwenjie
# @project : code
# @File    : BVcross_pssm_update.py
# @IDE     : PyCharm
# @REMARKS : 说明文字
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from os import path
import re
import Levenshtein
from tqdm import tqdm
from biopandas.pdb import PandasPdb
import heapq
from sklearn.cluster import KMeans
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=30,progress_bar=True)


df = pd.read_csv('../../../../data/sequence/features_related/BV_ALL_SCAN.csv')

df_A = df.loc[df['Chain'] == 'A'].copy()
df_C = df.loc[df['Chain'] == 'C'].copy()
df_E = df.loc[df['Chain'] == 'E'].copy()


df_A.reset_index(drop=True, inplace=True)
df_C.reset_index(drop=True, inplace=True)
df_E.reset_index(drop=True, inplace=True)


for df_j in [df_A,df_C,df_E]:
    df_j['residue_num'] = range(0,341)
    for i in range(df_j.shape[0]):

        if i+1 < 164:
            df_j.loc[i, 'site'] = int(i+1)
        elif i+1 == 164:
            df_j.loc[i, 'site'] = '163A'
        elif i+1 == 165:
            df_j.loc[i, 'site'] = '163B'
        elif i+1 == 166:
            df_j.loc[i, 'site'] = '163C'
        elif i+1 > 166:
            df_j.loc[i, 'site'] = i+1-3



with open(r'../../../../data/sequence/features_related/BV_RSA_freesasa.txt') as f:
    lines = f.readlines()


df_exposure = pd.DataFrame()
i = 0
for line in lines:
    content = line.split()
    # print(content)
    df_exposure.loc[i,'chain'] = content[2]
    # df_exposure.loc[i,'site'] = content[3]
    df_exposure.loc[i,'SASA'] = content[4]
    df_exposure.loc[i,'RSA'] = content[5]
    i+=1
df_exposure[['SASA']] = df_exposure[['SASA']].astype(float)
df_exposure[['RSA']] = df_exposure[['RSA']].astype(float)


# threshold = np.percentile(df_exposure['RSA'],20)
# print(threshold)
df_exposure.loc[(df_exposure['RSA'] >= 15),['exposed']] = 1
df_exposure.loc[(df_exposure['RSA'] < 15),['exposed']] = 0


exposure_A = df_exposure.loc[df_exposure['chain'] == 'A'].copy()
exposure_A.reset_index(drop=True, inplace=True)
df_A[['SASA','RSA','exposed']] = exposure_A[['SASA','RSA','exposed']]

exposure_C = df_exposure.loc[df_exposure['chain'] == 'C'].copy()
exposure_C.reset_index(drop=True, inplace=True)
df_C[['SASA','RSA','exposed']] = exposure_C[['SASA','RSA','exposed']]

exposure_E = df_exposure.loc[df_exposure['chain'] == 'E'].copy()
exposure_E.reset_index(drop=True, inplace=True)
df_E[['SASA','RSA','exposed']] = exposure_E[['SASA','RSA','exposed']]


df_A = df_A.sort_values(by='Binding site probability',ascending=False)
df_C = df_C.sort_values(by='Binding site probability',ascending=False)
df_E = df_E.sort_values(by='Binding site probability',ascending=False)


df_A.reset_index(drop=True, inplace=True)
df_C.reset_index(drop=True, inplace=True)
df_E.reset_index(drop=True, inplace=True)



epi_A = df_A.loc[:149,:].copy()
epi_C = df_C.loc[:149,:].copy()
epi_E = df_E.loc[:149,:].copy()
list_A = list(epi_A['residue_num'].loc[epi_A['exposed'] == 1])
list_C = list(epi_C['residue_num'].loc[epi_C['exposed'] == 1])
list_E = list(epi_E['residue_num'].loc[epi_E['exposed'] == 1])

BV_pdb = PandasPdb().read_pdb('../../../../data/sequence/features_related/BV_template.pdb')
df_BV = BV_pdb.df['ATOM'].copy()
axis_A = df_BV[(df_BV['chain_id'] == 'A') & (df_BV['atom_name'] == 'CA')].copy()
axis_C = df_BV[(df_BV['chain_id'] == 'C') & (df_BV['atom_name'] == 'CA')].copy()
axis_E = df_BV[(df_BV['chain_id'] == 'E') & (df_BV['atom_name'] == 'CA')].copy()
axis_A.reset_index(drop=True,inplace=True)
axis_C.reset_index(drop=True,inplace=True)
axis_E.reset_index(drop=True,inplace=True)



head_domain = list(range(51,281))
epiA_index = [i for i in list_A if i in head_domain]
epiC_index = [i for i in list_C if i in head_domain]
epiE_index = [i for i in list_E if i in head_domain]
# print(head_domain)
# print(list_A)
# print(epiA_index)
# print(list_C)
# print(epiC_index)
# print(list_E)
# print(epiE_index)


list_index = list(set(epiA_index).union(set(epiC_index),set(epiE_index)))
print(list_index)
print(len(list_index))


for list_j in [epiA_index,epiC_index,epiE_index]:
    for i in range(len(list_j)):
        if list_j[i] < 163:
            list_j[i] = list_j[i]+1
        elif list_j[i] == 163:
            list_j[i] = '163A'
        elif list_j[i] == 164:
            list_j[i] = '163B'
        elif list_j[i] == 165:
            list_j[i] = '163C'
        elif list_j[i] > 165:
            list_j[i] = list_j[i]+1-3


list_union = list(set(epiA_index).union(set(epiC_index),set(epiE_index)))

print('epiA_site:',epiA_index)
print('epiC_site:',epiC_index)
print('epiE_site:',epiE_index)


print(list_union)
print(len(list_union))



BV_pdb = PandasPdb().read_pdb('../../../../data/sequence/features_related/BV_template.pdb')
df_BV = BV_pdb.df['ATOM']
df_BV = df_BV[(df_BV['chain_id'] == 'A') & (df_BV['atom_name'] == 'CA')].copy()
df_BV.reset_index(drop=True,inplace=True)



estimator = KMeans(n_clusters=5,random_state=2022)   #构造聚类器
estimator.fit(df_BV.loc[list_index,['x_coord','y_coord','z_coord']])      #聚类
label_pred = estimator.labels_     #获取聚类标签
df_BV.loc[list_index,'label'] = label_pred
# df_BV.to_csv(r'/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BV_gasaid/BVseq_final/BVpre_epitope/test.csv')



epi_1 = df_BV[df_BV['label']==0]
epi_2 = df_BV[df_BV['label']==1]
epi_3 = df_BV[df_BV['label']==2]
epi_4 = df_BV[df_BV['label']==3]
epi_5 = df_BV[df_BV['label']==4]
# epi_6 = df_BV[df_BV['label']==5]
list_epi1 = list(epi_1.index)
list_epi2 = list(epi_2.index)
list_epi3 = list(epi_3.index)
list_epi4 = list(epi_4.index)
list_epi5 = list(epi_5.index)


list_aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


path = '../../../../data/sequence/features_related/BV.pssm'
with open(path, 'r') as f:
    lines = f.readlines()[3:-6]
    pssm = np.array([line.split()[2:22] for line in lines], dtype=int)

print(pssm.shape)

def pssm_diff(list_index, seq1, seq2):
    

    pssm_score = 0
    for index in list_index:

        if seq1[index] == seq2[index]:
            continue

        if seq1[index] != seq2[index]:

            if (seq1[index] != '-') & (seq2[index] != '-'):
                score = abs(pssm[index, list_aa.index(seq1[index])] - pssm[index, list_aa.index(seq2[index])])
                pssm_score = pssm_score + score
                continue

            if (seq1[index] == '-') & (seq2[index] != '-'):
                list_score = []
                for aa in list_aa:
                    aa_score = abs(pssm[index, list_aa.index(aa)] - pssm[index, list_aa.index(seq2[index])])
                    list_score.append(aa_score)
                score = max(list_score)
                pssm_score = pssm_score + score
                continue

            if (seq1[index] != '-') & (seq2[index] == '-'):
                list_score = []
                for aa in list_aa:
                    aa_score = abs(pssm[index, list_aa.index(seq1[index])] - pssm[index, list_aa.index(aa)])
                    list_score.append(aa_score)
                score = max(list_score)
                pssm_score = pssm_score + score
                continue

    return pssm_score

print(list_epi1)
print(list_epi2)
print(list_epi3)
print(list_epi4)
print(list_epi5)

data = pd.read_csv(r'../../../../data/result/BVHA1_predictdata_update.csv',index_col=0)


data['x_pssmepi1'] = data.parallel_apply(lambda row: pssm_diff(list_index=list_epi1,seq1=row['virus1_seq'],seq2=row['virus2_seq']),axis=1)
data['x_pssmepi2'] = data.parallel_apply(lambda row: pssm_diff(list_index=list_epi2,seq1=row['virus1_seq'],seq2=row['virus2_seq']),axis=1)
data['x_pssmepi3'] = data.parallel_apply(lambda row: pssm_diff(list_index=list_epi3,seq1=row['virus1_seq'],seq2=row['virus2_seq']),axis=1)
data['x_pssmepi4'] = data.parallel_apply(lambda row: pssm_diff(list_index=list_epi4,seq1=row['virus1_seq'],seq2=row['virus2_seq']),axis=1)
data['x_pssmepi5'] = data.parallel_apply(lambda row: pssm_diff(list_index=list_epi5,seq1=row['virus1_seq'],seq2=row['virus2_seq']),axis=1)


data.to_csv(r'../../../../data/result/BVHA1_predictdata_update.csv')


for list_j in [list_epi1,list_epi2,list_epi3,list_epi4,list_epi5]:
    for i in range(len(list_j)):
        if list_j[i] < 163:
            list_j[i] = list_j[i]+1
        elif list_j[i] == 163:
            list_j[i] = '163A'
        elif list_j[i] == 164:
            list_j[i] = '163B'
        elif list_j[i] == 165:
            list_j[i] = '163C'
        elif list_j[i] > 165:
            list_j[i] = list_j[i]+1-3

print(list_epi1)
print(list_epi2)
print(list_epi3)
print(list_epi4)
print(list_epi5)
