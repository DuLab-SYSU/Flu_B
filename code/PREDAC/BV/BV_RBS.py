# -*- coding: utf-8 -*-
# @Time    : 2023/12/22 下午3:07
# @Author  : Hanwenjie
# @project : code
# @File    : BV_RBS.py
# @IDE     : PyCharm
import matplotlib.pyplot as plt
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
# import swifter
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=30,progress_bar=True)


#import model dataset
df = pd.read_csv(r'../data/PREDAC/BV_model.csv',index_col=0)

def acid_diff(str1,str2):

    '''find the RBS site'''

    list_140loop = [i for i in range(136,144)]
    list_190helix = [i for i in range(196, 206)]
    list_240loop = [i for i in range(240, 246)]
    global list_rbs
    list_rbs = list_140loop + list_190helix +list_240loop
    list_diff = []

    for i in range(1,342):
        if str1[i-1] != str2[i-1]:
            list_diff.append(i)
    return list_diff

def calEuclidean(str1,str2):

    '''calculate the average Euclidean Distance'''

    list_eucfinal = []
    list_diff = acid_diff(str1,str2)
    if len(list_diff) == 0:
        euc_change = 0

    elif len(list_diff) != 0:
        for i in list_diff:
            list_region = list_rbs.copy()
            arr_str1 = np.array([df_BV.loc[i-1,'x_coord'],df_BV.loc[i-1,'y_coord'],df_BV.loc[i-1,'z_coord']]) #注意索引从0开始
            if i in list_region:
                list_region.remove(i)
            list_euc = []

            for j in list_region:
                arr_rbs = np.array([df_BV.loc[j-1,'x_coord'],df_BV.loc[j-1,'y_coord'],df_BV.loc[j-1,'z_coord']]) #第j个受体结合位点的坐标
                euc = np.linalg.norm(arr_str1-arr_rbs)
                list_euc.append(euc)
            list_eucfinal.append(min(list_euc))

        if len(list_eucfinal) <= 3:
            euc_change = sum(list_eucfinal)/len(list_eucfinal)
        elif len(list_eucfinal) > 3:
            list_max3 = heapq.nlargest(3,list_eucfinal)
            euc_change = sum(list_max3)/3

    return euc_change

aa_codes = {
     'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E',
     'PHE':'F', 'GLY':'G', 'HIS':'H', 'LYS':'K',
     'ILE':'I', 'LEU':'L', 'MET':'M', 'ASN':'N',
     'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S',
     'THR':'T', 'VAL':'V', 'TYR':'Y', 'TRP':'W'}

seq = ''
for line in open("../data/PREDAC/BV_template.pdb"):
    if line[0:6] == "SEQRES":
        columns = line.split()
        if columns[2] == 'A':
            for resname in columns[4:]:
                seq = seq + aa_codes[resname]
print(seq)
print(len(seq))


BV_pdb = PandasPdb().read_pdb('../data/PREDAC/BV_template.pdb')
df_BV = BV_pdb.df['ATOM']
df_BV = df_BV[(df_BV['chain_id'] == 'A') & (df_BV['atom_name'] == 'CA')]
df_BV.reset_index(drop=True,inplace=True)


df['x_rbs'] = df.parallel_apply(lambda row: calEuclidean(row['virus1_seq'],row['virus2_seq']),axis=1)

# export model dataset
df.to_csv(r'../data/PREDAC/BV_model.csv')