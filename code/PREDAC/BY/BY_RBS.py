# -*- coding: utf-8 -*-
# @Time    : 2023/12/24 下午2:58
# @Author  : Hanwenjie
# @project : code
# @File    : BY_RBS.py
# @IDE     : PyCharm
# @REMARKS : 说明文字
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

# Import sampled data based on num
df = pd.read_csv(r'/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BY_gasaid/BY_region_seq/data_process/final/model/BY_model.csv',index_col=0)

def acid_diff(str1,str2):

    '''Identify the receptor-binding sites that changed in BY subtype'''

    list_140loop = [i for i in range(136,144)]
    list_190helix = [i for i in range(195, 205)]
    list_240loop = [i for i in range(239, 245)]
    global list_rbs # Declare global variable
    list_rbs = list_140loop + list_190helix +list_240loop # All BY subtype receptor-binding sites
    list_diff = [] # Store the changed receptor-binding sites

    for i in range(1,344): # Note: this searches for different amino acid sites in the entire HA1 region, but due to missing residues in the template, it only goes up to 341
        if str1[i-1] != str2[i-1]: # Note: the i-th residue has index i-1 in the sequence
            list_diff.append(i)
    return list_diff

BY_pdb = PandasPdb().read_pdb('/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BY_gasaid/BYseq_final/BY_receptor/BY_template.pdb')
df_BY = BY_pdb.df['ATOM']
df_BY = df_BY[(df_BY['chain_id'] == 'A') & (df_BY['atom_name'] == 'CA')]
df_BY.reset_index(drop=True,inplace=True)
# df_BY.loc[:,'residue_number'] = [i for i in range(1,342)]
print(df_BY.index)
# df_BY.to_csv(r'/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BY_gasaid/BYseq_final/BY_receptor/test.csv')
print(df_BY.columns)
print(df_BY.shape[0])

def calEuclidean(str1,str2):

    '''Calculate the average Euclidean distance change between two strains'''

    list_eucfinal = [] # Store the shortest Euclidean distance between each changed amino acid site and the receptor-binding region
    list_diff = acid_diff(str1,str2) # Define changed amino acid sites
    # If there are no changed amino acid sites, return 0 directly
    if len(list_diff) == 0:
        euc_change = 0

    elif len(list_diff) != 0:
        # Iterate through each changed amino acid site and locate its coordinates
        for i in list_diff:
            list_region = list_rbs.copy()
            arr_str1 = np.array([df_BY.loc[i-1,'x_coord'],df_BY.loc[i-1,'y_coord'],df_BY.loc[i-1,'z_coord']]) # Note: index starts from 0
            if i in list_region:
                list_region.remove(i) # Define receptor-binding region excluding the current site
            list_euc = [] # Store Euclidean distances from this changed site to all receptor-binding sites
            # Calculate Euclidean distance between the changed site and each receptor-binding site
            for j in list_region:
                arr_rbs = np.array([df_BY.loc[j-1,'x_coord'],df_BY.loc[j-1,'y_coord'],df_BY.loc[j-1,'z_coord']]) # Coordinates of the j-th receptor-binding site
                euc = np.linalg.norm(arr_str1-arr_rbs)
                list_euc.append(euc)
            list_eucfinal.append(min(list_euc)) # Save the shortest distance for this site to the receptor-binding region
        # Calculate the average Euclidean distance for all changed sites; if more than 3, take the top 3 largest
        if len(list_eucfinal) <= 3:
            euc_change = sum(list_eucfinal)/len(list_eucfinal)
        elif len(list_eucfinal) > 3:
            list_max3 = heapq.nlargest(3,list_eucfinal)
            euc_change = sum(list_max3)/3

    return euc_change # Return the result

df['x_rbs'] = df.parallel_apply(lambda row: calEuclidean(row['virus1_seq'],row['virus2_seq']),axis=1)

# Export sampled data based on num
df.to_csv(r'/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BY_gasaid/BY_region_seq/data_process/final/model/BY_model.csv')

print(max(df['x_rbs']))
print(min(df['x_rbs']))