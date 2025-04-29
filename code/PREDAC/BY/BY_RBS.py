# -*- coding: utf-8 -*-
# @Time    : 2023/12/24 下午2:58
# @Author  : Hanwenjie
# @project : code
# @File    : BY_RBS.py
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
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=30,progress_bar=True)

#import model dataset
df = pd.read_csv(r'../data/PREDAC/BY_model.csv',index_col=0)

def acid_diff(str1,str2):

    '''find the RBS site'''

    list_140loop = [i for i in range(136,144)]
    list_190helix = [i for i in range(195, 205)]
    list_240loop = [i for i in range(239, 245)]
    global list_rbs
    list_rbs = list_140loop + list_190helix +list_240loop
    list_diff = []

    for i in range(1,344):
        if str1[i-1] != str2[i-1]:
            list_diff.append(i)
    return list_diff

BY_pdb = PandasPdb().read_pdb('../data/PREDAC/BY_template.pdb')
df_BY = BY_pdb.df['ATOM']
df_BY = df_BY[(df_BY['chain_id'] == 'A') & (df_BY['atom_name'] == 'CA')]
df_BY.reset_index(drop=True,inplace=True)

def calEuclidean(str1,str2):

    '''calculate the average Euclidean Distance'''

    list_eucfinal = []
    list_diff = acid_diff(str1,str2)

    if len(list_diff) == 0:
        euc_change = 0

    elif len(list_diff) != 0:

        for i in list_diff:
            list_region = list_rbs.copy()
            arr_str1 = np.array([df_BY.loc[i-1,'x_coord'],df_BY.loc[i-1,'y_coord'],df_BY.loc[i-1,'z_coord']]) #注意索引从0开始
            if i in list_region:
                list_region.remove(i)
            list_euc = []

            for j in list_region:
                arr_rbs = np.array([df_BY.loc[j-1,'x_coord'],df_BY.loc[j-1,'y_coord'],df_BY.loc[j-1,'z_coord']]) #第j个受体结合位点的坐标
                euc = np.linalg.norm(arr_str1-arr_rbs)
                list_euc.append(euc)
            list_eucfinal.append(min(list_euc))

        if len(list_eucfinal) <= 3:
            euc_change = sum(list_eucfinal)/len(list_eucfinal)
        elif len(list_eucfinal) > 3:
            list_max3 = heapq.nlargest(3,list_eucfinal)
            euc_change = sum(list_max3)/3

    return euc_change

df['x_rbs'] = df.parallel_apply(lambda row: calEuclidean(row['virus1_seq'],row['virus2_seq']),axis=1)

# export model datase
df.to_csv(r'../data/PREDAC/BY_model.csv')

