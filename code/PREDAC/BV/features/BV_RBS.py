# -*- coding: utf-8 -*-
# @Time    : 2023/12/22 3:07 PM
# @Author  : Hanwenjie
# @project : code
# @File    : BV_RBS.py
# @IDE     : PyCharm
# @REMARKS : Description
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


# Import HI median data
df = pd.read_csv(r'../../../data/BV_model.csv',index_col=0)

def acid_diff(str1,str2):

    '''Find receptor-binding sites of BV subtype variations'''

    list_140loop = [i for i in range(136,144)]
    list_190helix = [i for i in range(196, 206)]
    list_240loop = [i for i in range(240, 246)]
    global list_rbs # declare global variable
    list_rbs = list_140loop + list_190helix +list_240loop # all receptor-binding sites of BV subtype
    list_diff = [] # store mutated receptor-binding sites

    for i in range(1,342): # Note: search for amino acid differences across the entire HA1 region, but only up to 341 due to template deletion
        if str1[i-1] != str2[i-1]: # Note: the i-th position in the sequence corresponds to index i-1
            list_diff.append(i)
    return list_diff

def calEuclidean(str1,str2):

    '''Calculate the average Euclidean distance change between two strains'''

    list_eucfinal = [] # store the shortest Euclidean distance from each mutated amino acid site to the receptor-binding region
    list_diff = acid_diff(str1,str2) # define different amino acid positions
    # if no amino acid site changes, return 0
    if len(list_diff) == 0:
        euc_change = 0

    elif len(list_diff) != 0:
        # iterate through each mutated amino acid site and locate its coordinates
        for i in list_diff:
            list_region = list_rbs.copy()
            arr_str1 = np.array([df_BV.loc[i-1,'x_coord'],df_BV.loc[i-1,'y_coord'],df_BV.loc[i-1,'z_coord']]) # Note: index starts from 0
            if i in list_region:
                list_region.remove(i) # define receptor-binding region
            list_euc = [] # store Euclidean distances from one mutated site to each receptor-binding site
            # calculate Euclidean distances between each mutated site and each receptor-binding site
            for j in list_region:
                arr_rbs = np.array([df_BV.loc[j-1,'x_coord'],df_BV.loc[j-1,'y_coord'],df_BV.loc[j-1,'z_coord']]) # coordinates of the j-th receptor-binding site
                euc = np.linalg.norm(arr_str1-arr_rbs)
                list_euc.append(euc)
            list_eucfinal.append(min(list_euc)) # save the shortest Euclidean distance to the receptor-binding region
        # calculate the average Euclidean distance of all mutated sites;
        # if more than 3 mutated sites, take the top 3
        if len(list_eucfinal) <= 3:
            euc_change = sum(list_eucfinal)/len(list_eucfinal)
        elif len(list_eucfinal) > 3:
            list_max3 = heapq.nlargest(3,list_eucfinal)
            euc_change = sum(list_max3)/3

    return euc_change # return result

aa_codes = {
     'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E',
     'PHE':'F', 'GLY':'G', 'HIS':'H', 'LYS':'K',
     'ILE':'I', 'LEU':'L', 'MET':'M', 'ASN':'N',
     'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S',
     'THR':'T', 'VAL':'V', 'TYR':'Y', 'TRP':'W'}

seq = ''
for line in open("../../../data/BV_template.pdb"):
    if line[0:6] == "SEQRES":
        columns = line.split()
        if columns[2] == 'A':
            for resname in columns[4:]:
                seq = seq + aa_codes[resname]
print(seq)
print(len(seq))


BV_pdb = PandasPdb().read_pdb('../../../data/BV_template.pdb')
df_BV = BV_pdb.df['ATOM']
df_BV = df_BV[(df_BV['chain_id'] == 'A') & (df_BV['atom_name'] == 'CA')]
df_BV.reset_index(drop=True,inplace=True)
# df_BV.loc[:,'residue_number'] = [i for i in range(1,342)]
print(df_BV.index)



df['x_rbs'] = df.parallel_apply(lambda row: calEuclidean(row['virus1_seq'],row['virus2_seq']),axis=1)

# Export HI median data
df.to_csv(r'./result/BV_model.csv')
