# -*- coding: utf-8 -*-
# @Time    : 2023/12/25 下午9:11
# @Author  : Hanwenjie
# @project : code
# @File    : BYcross_RBS.py
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

#导入根据num采样数据
df = pd.read_csv(r'../../../../data/result/BYHA1_predictdata.csv',index_col=0)

def acid_diff(str1,str2):

    '''找出BY亚型变换的受体结合位点'''

    list_140loop = [i for i in range(136,144)]
    list_190helix = [i for i in range(195, 205)]
    list_240loop = [i for i in range(239, 245)]
    global list_rbs #声明全局变量
    list_rbs = list_140loop + list_190helix +list_240loop #所有BY亚型受体结合位点
    list_diff = [] #储存变换的受体结合位点

    for i in range(1,344): #注意此部分在整个HA1区域寻找不同的氨基酸位点，但是由于模板中有缺失，因此只到341
        if str1[i-1] != str2[i-1]: #注意序类的第i个位点索引为i-1
            list_diff.append(i)
    return list_diff

BY_pdb = PandasPdb().read_pdb('../../../../data/sequence/features_related/BY_template.pdb')
df_BY = BY_pdb.df['ATOM']
df_BY = df_BY[(df_BY['chain_id'] == 'A') & (df_BY['atom_name'] == 'CA')]
df_BY.reset_index(drop=True,inplace=True)
# df_BY.loc[:,'residue_number'] = [i for i in range(1,342)]
print(df_BY.index)
# df_BY.to_csv(r'/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BY_gasaid/BYseq_final/BY_receptor/test.csv')
print(df_BY.columns)
print(df_BY.shape[0])

def calEuclidean(str1,str2):

    '''求两个毒株之间的平均欧式距离变化'''

    list_eucfinal = [] #用于存储每个变换的氨基酸位点与受体结合区域的最短欧式距离
    list_diff = acid_diff(str1,str2) #定义不同的位点不同的氨基酸位置
    #如果没有变化的氨基酸位点，则直接返回0
    if len(list_diff) == 0:
        euc_change = 0

    elif len(list_diff) != 0:
        #遍历每个变换的氨基酸位点，并定位其坐标
        for i in list_diff:
            list_region = list_rbs.copy()
            arr_str1 = np.array([df_BY.loc[i-1,'x_coord'],df_BY.loc[i-1,'y_coord'],df_BY.loc[i-1,'z_coord']]) #注意索引从0开始
            if i in list_region:
                list_region.remove(i) #定义受体结合位点区域
            list_euc = [] #用于存储某个变化位点与受体结合区域每个位点的欧式距离
            #计算每个变化的氨基酸位点与受体结合区域每个位点的欧式距离
            for j in list_region:
                arr_rbs = np.array([df_BY.loc[j-1,'x_coord'],df_BY.loc[j-1,'y_coord'],df_BY.loc[j-1,'z_coord']]) #第j个受体结合位点的坐标
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

df['x_rbs'] = df.parallel_apply(lambda row: calEuclidean(row['virus1_seq'],row['virus2_seq']),axis=1)

#导出num采样数据
df.to_csv(r'../../../../data/result/BYHA1_predictdata.csv')

print(max(df['x_rbs']))
print(min(df['x_rbs']))