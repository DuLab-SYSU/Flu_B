# -*- coding: utf-8 -*-
# @Time    : 2024/1/8 上午11:40
# @Author  : Hanwenjie
# @project : code
# @File    : BV_crosstable_update.py
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
from Bio import SeqIO
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm
import time


df = pd.read_csv(r'../../../../data/sequence/BV_sample_final.csv',index_col=0)
df.rename(columns={'seq':'seq_selfill'},inplace=True)
df = df[df['collection_year'] >= 1987]
df.reset_index(inplace=True, drop=True)
print(df.columns)

seq = {}
name = {}
num = {}
region = {}
year = {}
for i in range(df.shape[0]):
    seq[df.loc[i, 'description']] = df.loc[i, 'seq_selfill']
    name[df.loc[i, 'description']] = df.loc[i, 'virus']
    num[df.loc[i, 'description']] = df.loc[i, 'num']
    region[df.loc[i, 'description']] = df.loc[i, 'region']
    year[df.loc[i, 'description']] = df.loc[i, 'collection_year']

# print(seq)
# print(name)

des_virus1 = []
des_virus2 = []
seq_virus1 = []
seq_virus2 = []
name_virus1 = []
name_virus2 = []
num_virus1 = []
num_virus2 = []
region_virus1 = []
region_virus2 = []
year_virus1 = []
year_virus2 = []
name_list = list(seq.keys())
n = 0
for i in tqdm(range(len(name_list))):
    for j in range(i + 1, len(name_list)):
        des_virus1.append(name_list[i])
        des_virus2.append(name_list[j])
        seq_virus1.append(seq[des_virus1[-1]])
        seq_virus2.append(seq[des_virus2[-1]])
        name_virus1.append(name[des_virus1[-1]])
        name_virus2.append(name[des_virus2[-1]])
        region_virus1.append(region[des_virus1[-1]])
        region_virus2.append(region[des_virus2[-1]])
        num_virus1.append(num[des_virus1[-1]])
        num_virus2.append(num[des_virus2[-1]])
        year_virus1.append(year[des_virus1[-1]])
        year_virus2.append(year[des_virus2[-1]])
        n += 1
print(n)

data = {'virus_1': name_virus1, 'virus_2': name_virus2, 'virus1_description': des_virus1,
        'virus2_description': des_virus2, 'virus1_seq': seq_virus1,
        'virus2_seq': seq_virus2, 'virus1_num': num_virus1, 'virus2_num': num_virus2,
        'virus1_region': region_virus1, 'virus2_region': region_virus2,
        'virus1_year': year_virus1, 'virus2_year': year_virus2, }
df_predict = pd.DataFrame(data)

df_predict.to_csv(r'../../../../data/result/predictdata/BVHA1_predictdata_update.csv')

