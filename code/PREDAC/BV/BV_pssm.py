# -*- coding: utf-8 -*-
# @Time    : 2023/12/22 下午2:52
# @Author  : Hanwenjie
# @project : code
# @File    : BV_pssm.py
# @IDE     : PyCharm

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

#import predicted antigenic probability of each site by ScanNet
df = pd.read_csv('../data/PREDAC/BV_ALL_SCAN.csv')

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

#import the exposed area of each site by FreeSASA
with open(r'../data/PREDAC/BV_RSA_freesasa.txt') as f:
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

BV_pdb = PandasPdb().read_pdb('../data/PREDAC/BV_template.pdb')
df_BV = BV_pdb.df['ATOM'].copy()
axis_A = df_BV[(df_BV['chain_id'] == 'A') & (df_BV['atom_name'] == 'CA')].copy()
axis_C = df_BV[(df_BV['chain_id'] == 'C') & (df_BV['atom_name'] == 'CA')].copy()
axis_E = df_BV[(df_BV['chain_id'] == 'E') & (df_BV['atom_name'] == 'CA')].copy()
axis_A.reset_index(drop=True,inplace=True)
axis_C.reset_index(drop=True,inplace=True)
axis_E.reset_index(drop=True,inplace=True)


#head domain area of HA segment
head_domain = list(range(51,281))
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
        elif list_j[i] == 165:
            list_j[i] = '163C'
        elif list_j[i] > 165:
            list_j[i] = list_j[i]+1-3


list_union = list(set(epiA_index).union(set(epiC_index),set(epiE_index)))

BV_pdb = PandasPdb().read_pdb('../data/PREDAC/BV_template.pdb')
df_BV = BV_pdb.df['ATOM']
df_BV = df_BV[(df_BV['chain_id'] == 'A') & (df_BV['atom_name'] == 'CA')].copy()
df_BV.reset_index(drop=True,inplace=True)

# cluster
estimator = KMeans(n_clusters=5,random_state=2022)
estimator.fit(df_BV.loc[list_index,['x_coord','y_coord','z_coord']])
label_pred = estimator.labels_
df_BV.loc[list_index,'label'] = label_pred

# defined epitopes
epi_1 = df_BV[df_BV['label']==0]
epi_2 = df_BV[df_BV['label']==1]
epi_3 = df_BV[df_BV['label']==2]
epi_4 = df_BV[df_BV['label']==3]
epi_5 = df_BV[df_BV['label']==4]

list_epi1 = list(epi_1.index)
list_epi2 = list(epi_2.index)
list_epi3 = list(epi_3.index)
list_epi4 = list(epi_4.index)
list_epi5 = list(epi_5.index)

# index for PSSM
list_aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# import PSSM matrix
path = '../data/PREDAC/BV.pssm'
with open(path, 'r') as f:
    lines = f.readlines()[3:-6]
    pssm = np.array([line.split()[2:22] for line in lines], dtype=int)

print(pssm.shape)


def pssm_diff(list_index, seq1, seq2):
    '''calculate the sum score of PSSM for each epitope'''
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

#import model dataset
data = pd.read_csv(r'../data/PREDAC/BV_model.csv',index_col=0)

#calculate epitope-PSSM feature
data['x_pssmepi1'] = data.parallel_apply(lambda row: pssm_diff(list_index=list_epi1,seq1=row['virus1_seq'],seq2=row['virus2_seq']),axis=1)
data['x_pssmepi2'] = data.parallel_apply(lambda row: pssm_diff(list_index=list_epi2,seq1=row['virus1_seq'],seq2=row['virus2_seq']),axis=1)
data['x_pssmepi3'] = data.parallel_apply(lambda row: pssm_diff(list_index=list_epi3,seq1=row['virus1_seq'],seq2=row['virus2_seq']),axis=1)
data['x_pssmepi4'] = data.parallel_apply(lambda row: pssm_diff(list_index=list_epi4,seq1=row['virus1_seq'],seq2=row['virus2_seq']),axis=1)
data['x_pssmepi5'] = data.parallel_apply(lambda row: pssm_diff(list_index=list_epi5,seq1=row['virus1_seq'],seq2=row['virus2_seq']),axis=1)

#export model dataset
data.to_csv(r'../data/PREDAC/BV_model.csv')

