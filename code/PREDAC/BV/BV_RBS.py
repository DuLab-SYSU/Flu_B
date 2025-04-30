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

    '''We will upload the complete code here once the manuscript is officially published'''
    return list_diff

def calEuclidean(str1,str2):

    '''calculate the average Euclidean Distance'''

    '''We will upload the complete code here once the manuscript is officially published'''

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
