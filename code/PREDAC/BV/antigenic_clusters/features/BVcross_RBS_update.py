# -*- coding: utf-8 -*-
# @Time    : 2024/1/8 上午11:46
# @Author  : Hanwenjie
# @project : code
# @File    : BVcross_RBS_update.py
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
# import swifter
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=30,progress_bar=True)

#导入HI取中位数数据
df = pd.read_csv(r'../../../../data/result/BVHA1_predictdata_update.csv',index_col=0)

def acid_diff(str1,str2):

    '''找出BV亚型变换的受体结合位点'''

    list_140loop = [i for i in range(136,144)]
    list_190helix = [i for i in range(196, 206)]
    list_240loop = [i for i in range(240, 246)]
    global list_rbs #声明全局变量
    list_rbs = list_140loop + list_190helix +list_240loop #所有BV亚型受体结合位点
    list_diff = [] #储存变换的受体结合位点

    for i in range(1,342): #注意此部分在整个HA1区域寻找不同的氨基酸位点，但是由于模板中有缺失，因此只到341
        if str1[i-1] != str2[i-1]: #注意序类的第i个位点索引为i-1
            list_diff.append(i)
    return list_diff

def calEuclidean(str1,str2):

    '''求两个毒株之间的平均欧式距离变化'''

    # list_140loop = [i for i in range(136, 144)]
    # list_190helix = [i for i in range(196, 206)]
    # list_240loop = [i for i in range(240, 246)]
    # list_rbs = list_140loop + list_190helix + list_240loop  # 所有BV亚型受体结合位点
    list_eucfinal = [] #用于存储每个变换的氨基酸位点与受体结合区域的最短欧式距离
    list_diff = acid_diff(str1,str2) #定义不同的位点不同的氨基酸位置
    #如果没有变化的氨基酸位点，则直接返回0
    if len(list_diff) == 0:
        euc_change = 0

    elif len(list_diff) != 0:
        #遍历每个变换的氨基酸位点，并定位其坐标
        for i in list_diff:
            list_region = list_rbs.copy()
            arr_str1 = np.array([df_BV.loc[i-1,'x_coord'],df_BV.loc[i-1,'y_coord'],df_BV.loc[i-1,'z_coord']]) #注意索引从0开始
            if i in list_region:
                list_region.remove(i) #定义受体结合位点区域
            list_euc = [] #用于存储某个变化位点与受体结合区域每个位点的欧式距离
            #计算每个变化的氨基酸位点与受体结合区域每个位点的欧式距离
            for j in list_region:
                arr_rbs = np.array([df_BV.loc[j-1,'x_coord'],df_BV.loc[j-1,'y_coord'],df_BV.loc[j-1,'z_coord']]) #第j个受体结合位点的坐标
                euc = np.linalg.norm(arr_str1-arr_rbs)
                list_euc.append(euc)
            list_eucfinal.append(min(list_euc)) #保存每个位点与受体结合区域的最短欧式距离
        #计算所有变化位点的平均欧式距离，如果变化位点大于3个，则取前三个
        if len(list_eucfinal) <= 3:
            euc_change = sum(list_eucfinal)/len(list_eucfinal)
        elif len(list_eucfinal) > 3:
            list_max3 = heapq.nlargest(3,list_eucfinal)
            euc_change = sum(list_max3)/3

    return euc_change #返回结果

aa_codes = {
     'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E',
     'PHE':'F', 'GLY':'G', 'HIS':'H', 'LYS':'K',
     'ILE':'I', 'LEU':'L', 'MET':'M', 'ASN':'N',
     'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S',
     'THR':'T', 'VAL':'V', 'TYR':'Y', 'TRP':'W'}

seq = ''
for line in open("../../../../data/sequence/features_related/BV_template.pdb"):
    if line[0:6] == "SEQRES":
        columns = line.split()
        if columns[2] == 'A':
            for resname in columns[4:]:
                seq = seq + aa_codes[resname]
print(seq)
print(len(seq))


BV_pdb = PandasPdb().read_pdb('../../../../data/sequence/features_related/BV_template.pdb')
df_BV = BV_pdb.df['ATOM']
df_BV = df_BV[(df_BV['chain_id'] == 'A') & (df_BV['atom_name'] == 'CA')]
df_BV.reset_index(drop=True,inplace=True)
# df_BV.loc[:,'residue_number'] = [i for i in range(1,342)]
print(df_BV.index)
# df_BV.to_csv(r'/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BV_gasaid/BVseq_final/BV_receptor/test.csv')
# print(df_BV.columns)


# print(acid_diff(df.loc[4,'virus1_seq'],df.loc[4,'virus2_seq']))
# calEuclidean(df.loc[4,'virus1_seq'],df.loc[4,'virus2_seq'])
# print(calEuclidean(df.loc[0,'virus1_seq'],df.loc[0,'virus2_seq']))

df['x_rbs'] = df.parallel_apply(lambda row: calEuclidean(row['virus1_seq'],row['virus2_seq']),axis=1)

#导出HI取中位数数据
df.to_csv(r'../../../../data/result/BVHA1_predictdata_update.csv')


print(max(df['x_rbs']))
print(min(df['x_rbs']))