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



# define the num of different N-Gly sites of each two strains
def gly_diff(vir1,vir2):
	''' We will upload the complete code here once the manuscript is officially published.'''


#import model data
data = pd.read_csv(r'../data/PREDAC/BV_model.csv',index_col=0)
print(data.columns)
#import the number of seq
df_seq = pd.read_csv(r'../data/PREDAC/BV_sample.csv',index_col=0)

df['x_Nglycosylation'] = df.parallel_apply(lambda row: gly_diff(row['virus1_num'],row['virus2_num']),axis=1)
print(df['x_Nglycosylation'].value_counts())

# export model data
df.to_csv(r'../data/PREDAC/BV_model.csv')
