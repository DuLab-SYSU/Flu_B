# -*- coding: utf-8 -*-
# @Time    : 2023/12/24 上午11:47
# @Author  : Hanwenjie
# @project : code
# @File    : BY_pssm.py
# @IDE     : PyCharm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import pandas as pd
import math
import numpy as np
import os
from os import path
import re
import Levenshtein
from tqdm import tqdm
from biopandas.pdb import PandasPdb
import heapq
from sklearn.cluster import KMeans
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=20,progress_bar=True)
#import predicted antigenic probability of each site by ScanNet
df = pd.read_csv('../data/PREDAC/BY_ALL_SCAN.csv')

#split by chain
df_A = df.loc[df['Chain'] == 'A'].copy()
df_C = df.loc[df['Chain'] == 'C'].copy()
df_E = df.loc[df['Chain'] == 'E'].copy()

#reset index
#change the deletion location
df_A.reset_index(drop=True, inplace=True)
df_C.reset_index(drop=True, inplace=True)
df_E.reset_index(drop=True, inplace=True)

#reset site number
for df_j in [df_A,df_C,df_E]:
    df_j['residue_num'] = range(0,343)
    for i in range(df_j.shape[0]):

        if i+1 < 164:
            df_j.loc[i, 'site'] = int(i+1)
        elif i+1 == 164:
            df_j.loc[i, 'site'] = '163A'
        elif i+1 == 165:
            df_j.loc[i, 'site'] = '163B'
        elif i+1 > 165:
            df_j.loc[i, 'site'] = i+1-2

#import the exposed area of each site by FreeSASA
with open(r'../data/PREDAC/BY_RSA_freesasa.txt') as f:
    lines = f.readlines()

#transfer to dataframe
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

# the site is exposed if RSA >= 15%
df_exposure.loc[(df_exposure['RSA'] >= 15),['exposed']] = 1
df_exposure.loc[(df_exposure['RSA'] < 15),['exposed']] = 0

#combine antigenic and exposed dataframe
exposure_A = df_exposure.loc[df_exposure['chain'] == 'A'].copy()
exposure_A.reset_index(drop=True, inplace=True)
df_A[['SASA','RSA','exposed']] = exposure_A[['SASA','RSA','exposed']]

exposure_C = df_exposure.loc[df_exposure['chain'] == 'C'].copy()
exposure_C.reset_index(drop=True, inplace=True)
df_C[['SASA','RSA','exposed']] = exposure_C[['SASA','RSA','exposed']]

exposure_E = df_exposure.loc[df_exposure['chain'] == 'E'].copy()
exposure_E.reset_index(drop=True, inplace=True)
df_E[['SASA','RSA','exposed']] = exposure_E[['SASA','RSA','exposed']]

# sort by antigenic probability
df_A = df_A.sort_values(by='Binding site probability',ascending=False)
df_C = df_C.sort_values(by='Binding site probability',ascending=False)
df_E = df_E.sort_values(by='Binding site probability',ascending=False)

#reset index
df_A.reset_index(drop=True, inplace=True)
df_C.reset_index(drop=True, inplace=True)
df_E.reset_index(drop=True, inplace=True)

#extract top 150 antigenic probability and exposed site
epi_A = df_A.loc[:149,:].copy()
epi_C = df_C.loc[:149,:].copy()
epi_E = df_E.loc[:149,:].copy()
list_A = list(epi_A['residue_num'].loc[epi_A['exposed'] == 1])
list_C = list(epi_C['residue_num'].loc[epi_C['exposed'] == 1])
list_E = list(epi_E['residue_num'].loc[epi_E['exposed'] == 1])

BY_pdb = PandasPdb().read_pdb('../data/PREDAC/BY_template.pdb')
df_BY = BY_pdb.df['ATOM'].copy()
axis_A = df_BY[(df_BY['chain_id'] == 'A') & (df_BY['atom_name'] == 'CA')].copy()
axis_C = df_BY[(df_BY['chain_id'] == 'C') & (df_BY['atom_name'] == 'CA')].copy()
axis_E = df_BY[(df_BY['chain_id'] == 'E') & (df_BY['atom_name'] == 'CA')].copy()
axis_A.reset_index(drop=True,inplace=True)
axis_C.reset_index(drop=True,inplace=True)
axis_E.reset_index(drop=True,inplace=True)

#head domain area of HA segment
head_domain = list(range(51,280))
epiA_index = [i for i in list_A if i in head_domain]
epiC_index = [i for i in list_C if i in head_domain]
epiE_index = [i for i in list_E if i in head_domain]

list_index = list(set(epiA_index).union(set(epiC_index),set(epiE_index)))
print(list_index)
print(len(list_index))

#change the site number to real location
for list_j in [epiA_index,epiC_index,epiE_index]:
    for i in range(len(list_j)):
        if list_j[i] < 163:
            list_j[i] = list_j[i]+1
        elif list_j[i] == 163:
            list_j[i] = '163A'
        elif list_j[i] == 164:
            list_j[i] = '163B'
        elif list_j[i] > 164:
            list_j[i] = list_j[i]+1-2


list_union = list(set(epiA_index).union(set(epiC_index),set(epiE_index)))

print('epiA_site:',epiA_index)
print('epiC_site:',epiC_index)
print('epiE_site:',epiE_index)

print(list_union)
print(len(list_union))



BY_pdb = PandasPdb().read_pdb('../data/PREDAC/BY_template.pdb')
df_BY = BY_pdb.df['ATOM']
df_BY = df_BY[(df_BY['chain_id'] == 'A') & (df_BY['atom_name'] == 'CA')].copy()
df_BY.reset_index(drop=True,inplace=True)

# cluster
estimator = KMeans(n_clusters=5,random_state=2022)
estimator.fit(df_BY.loc[list_index,['x_coord','y_coord','z_coord']])
label_pred = estimator.labels_
df_BY.loc[list_index,'label'] = label_pred

# defined epitopes
epi_1 = df_BY[df_BY['label']==0]
epi_2 = df_BY[df_BY['label']==1]
epi_3 = df_BY[df_BY['label']==2]
epi_4 = df_BY[df_BY['label']==3]
epi_5 = df_BY[df_BY['label']==4]

list_epi1 = list(epi_1.index)
list_epi2 = list(epi_2.index)
list_epi3 = list(epi_3.index)
list_epi4 = list(epi_4.index)
list_epi5 = list(epi_5.index)


# 用于定位pssm矩阵的列索引
list_aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# import PSSM matrix
path = '../data/PREDAC/BY.pssm'
with open(path, 'r') as f:
    lines = f.readlines()[3:-6]
    pssm = np.array([line.split()[2:22] for line in lines], dtype=int)

print(pssm.shape)

def pssm_diff(list_index, seq1, seq2):
    '''calculate the sum score of PSSM for each epitope'''

    '''We will upload the complete code here once the manuscript is officially published'''

    return pssm_score

#import model dataset
data = pd.read_csv(r'../data/PREDAC/BY_model.csv',index_col=0)

#calculate epitope-PSSM featur
data['x_pssmepi1'] = data.parallel_apply(lambda row: pssm_diff(list_index=list_epi1,seq1=row['virus1_seq'],seq2=row['virus2_seq']),axis=1)
data['x_pssmepi2'] = data.parallel_apply(lambda row: pssm_diff(list_index=list_epi2,seq1=row['virus1_seq'],seq2=row['virus2_seq']),axis=1)
data['x_pssmepi3'] = data.parallel_apply(lambda row: pssm_diff(list_index=list_epi3,seq1=row['virus1_seq'],seq2=row['virus2_seq']),axis=1)
data['x_pssmepi4'] = data.parallel_apply(lambda row: pssm_diff(list_index=list_epi4,seq1=row['virus1_seq'],seq2=row['virus2_seq']),axis=1)
data['x_pssmepi5'] = data.parallel_apply(lambda row: pssm_diff(list_index=list_epi5,seq1=row['virus1_seq'],seq2=row['virus2_seq']),axis=1)

print(list_epi1)
print(list_epi2)
print(list_epi3)
print(list_epi4)
print(list_epi5)

#export model dataset
data.to_csv(r'../data/PREDAC/BY_model.csv')
