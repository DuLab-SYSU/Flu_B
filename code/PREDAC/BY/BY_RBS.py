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
    '''We will upload the complete code here once the manuscript is officially published'''
    return list_diff

BY_pdb = PandasPdb().read_pdb('../data/PREDAC/BY_template.pdb')
df_BY = BY_pdb.df['ATOM']
df_BY = df_BY[(df_BY['chain_id'] == 'A') & (df_BY['atom_name'] == 'CA')]
df_BY.reset_index(drop=True,inplace=True)

def calEuclidean(str1,str2):

    '''calculate the average Euclidean Distance'''
    '''We will upload the complete code here once the manuscript is officially published'''
    return euc_change

df['x_rbs'] = df.parallel_apply(lambda row: calEuclidean(row['virus1_seq'],row['virus2_seq']),axis=1)

# export model datase
df.to_csv(r'../data/PREDAC/BY_model.csv')

