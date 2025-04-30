# -*- coding: utf-8 -*-
# @Time    : 2023/12/22 下午4:16
# @Author  : Hanwenjie
# @project : code
# @File    : BV_Ngly.py
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
from pandarallel import pandarallel
from Bio import SeqIO
pandarallel.initialize(nb_workers=20, progress_bar=True)

# import sample data
df_virus = pd.read_csv(r'../data/PREDAC/BV_sample.csv',index_col=0)

with open(r'../data/PREDAC/BVHA1_Ngly.txt') as f:
	lines = f.readlines()

df_gly = pd.DataFrame()
i = 0
for line in lines:
	if len(re.findall(r'[+]',line.rstrip())) != 0:
		content = line.rstrip().split()
		# print(content)
		df_gly.loc[i,'num'] = content[0]
		df_gly.loc[i,'gly_position'] = content[1]
		i+=1

df_gly = df_gly.groupby('num')

vir_dict = {}
for group in df_gly:
	gly_list = list(group[1].gly_position)
	vir_dict[group[0]] = gly_list

print(len(vir_dict))

dict_copy1 = vir_dict.copy()
for key in dict_copy1.keys():
    if len(str(key)) > 4:
        del vir_dict[key]

dict_copy1 = vir_dict.copy()
for num, pos in dict_copy1.items():
    orign_new = {}
    seq = df_virus.loc[int(num)-1,'seq']
    n_gap = 0
    for i in range(len(seq)):
        if seq[i] == '-':
            n_gap+=1
        if seq[i] != '-':
            orign_new[str(i+1)] = str(i+1-n_gap)

    new_orign = dict([val, key] for key, val in orign_new.items())
    orign_pos = [new_orign[i] for i in pos]
    vir_dict[num] = orign_pos

# define the num of different N-Gly sites of each two strains
def gly_diff(vir1,vir2):
    '''We will upload the complete code here once the manuscript is officially published'''
    return num_diff

#import model data
data = pd.read_csv(r'../data/PREDAC/BV_model.csv',index_col=0)
print(data.columns)
#import the number of seq
df_seq = pd.read_csv(r'../data/PREDAC/BV_sample.csv',index_col=0)

#modify the name of columns
df_seq.rename(columns={'seq':'virus1_seq','num':'virus1_num'},inplace=True)
print(df_seq)
#map num for virus1
df = pd.merge(data,df_seq,on='virus1_seq')
print(df.shape[0])

#modify the name of columns
df_seq.rename(columns={'virus1_seq':'virus2_seq','virus1_num':'virus2_num'},inplace=True)
print(df_seq)
#map num for virus2
df = pd.merge(df,df_seq,on='virus2_seq')
print(df.shape[0])
df.reset_index(inplace=True,drop=True)
print(df.columns)
# transfer data type
df['virus1_num'] = df['virus1_num'].astype('str')
df['virus2_num'] = df['virus2_num'].astype('str')

df['x_Nglycosylation'] = df.parallel_apply(lambda row: gly_diff(row['virus1_num'],row['virus2_num']),axis=1)
print(df['x_Nglycosylation'].value_counts())

# export model data
df.to_csv(r'../data/PREDAC/BV_model.csv')
