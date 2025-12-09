# -*- coding: utf-8 -*-
# @Time    : 2024/1/11 下午5:19
# @Author  : Hanwenjie
# @project : code
# @File    : BV_transite_process.py
# @IDE     : PyCharm
# @REMARKS : 说明文字
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from matplotlib.transforms import ScaledTranslation
mpl.rcParams['pdf.fonttype']=42
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=30,progress_bar=True)

# Define the function for calculating entropy
def ent(data):
    prob1 = pd.value_counts(data) / len(data)
    return sum(np.log2(prob1) * prob1 * (-1))


# Define the function for calculating information gain
def gain(data, str1, str2):
    e1 = data.groupby(str1).apply(lambda x: ent(x[str2]))
    p1 = pd.value_counts(data[str1]) / len(data[str1])
    e2 = sum(e1 * p1)
    return ent(data[str2]) - e2


# Extract the amino acids at each position of HA1
df = pd.read_csv(r'../../../data/result/BV_cluster.csv',index_col=0)
print(df.shape[0])
print(df.columns)
df = pd.concat([df, df.seq.str.split('', expand=True).iloc[:, 1:-1]], axis=1)
pos_dict = {}
for i in range(1,347):
    pos_dict[i] = 'pos_' + str(i)
print(pos_dict)
df.rename(columns=pos_dict,inplace=True)
df.to_csv(r'../../../data/result/BV_IG.csv')

'''
# Extract the dataset of key loci for conversion between different antigenic clusters for plotting
df = pd.read_csv(r'../../../data/result/BV_IG.csv',index_col=0)
df_1_5 = df.loc[(df['cluster']==6) | (df['cluster']==2)].copy()
df_1_5.reset_index(inplace=True,drop=True)
# df_1_5 = df_1_5[['cluster','pos_164','pos_165','pos_166','pos_129']]

list_gain = []
list_ent = []
df_gain = pd.DataFrame()
df_gain['position'] = range(1,347)
for i in tqdm(range(1,347)):
    position = 'pos_' + str(i)
    pos_gain = 'pos' + str(i) + '_' + 'gain'
    pos_ent = 'pos' + str(i) + '_' + 'ent'
    df_gain.loc[i-1,'gain'] = gain(df_1_5,position,'cluster')
    df_gain.loc[i-1,'entropy'] = ent(df_1_5[position].copy())
    list_gain.append(gain(df_1_5,position,'cluster'))
    list_ent.append(ent(df_1_5[position].copy()))


max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
df_gain['gain_norm'] = df_gain[['gain']].apply(max_min_scaler)
df_gain['entropy_norm'] = df_gain[['entropy']].apply(max_min_scaler)

print(df_gain['gain_norm'].max())

print(df_gain['entropy_norm'].max())
df_gain.to_csv(r'../../../data/result/BV_IG_6_2.csv')
'''