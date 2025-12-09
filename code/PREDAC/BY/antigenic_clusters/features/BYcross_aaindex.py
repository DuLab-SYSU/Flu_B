# -*- coding: utf-8 -*-
# @Time    : 2023/12/25 下午9:09
# @Author  : Hanwenjie
# @project : code
# @File    : BYcross_aaindex.py
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
pandarallel.initialize(nb_workers=20,progress_bar=True)
import pandas as pd
from aaindex import aaindex1
import time
from tqdm import tqdm
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=22,progress_bar=True)

def AAindex(str1,str2):
    '''aaindex'''
    list_aaindex = []
    for s1, s2 in zip(str1, str2):

        if s1 != s2:
            if (s1 != '-') & (s2 != '-'):
                aaindex_change = abs(aaindex_dict[s1]-aaindex_dict[s2])
                list_aaindex.append(aaindex_change)
                continue

            if (s1 == '-') & (s2 != '-'):
                list_gap1 = []
                for value in aaindex_dict.values():
                    list_gap1.append(abs(value-aaindex_dict[s2]))
                list_aaindex.append(max(list_gap1))
                continue

            if (s1 != '-') & (s2 == '-'):
                list_gap2 = []
                for value in aaindex_dict.values():
                    list_gap2.append(abs(aaindex_dict[s1]-value))
                list_aaindex.append(max(list_gap2))
                continue
        else:
            pass

    if len(list_aaindex) == 0:
        ave_aaindex = 0
    elif len(list_aaindex) <= 3:
         ave_aaindex = sum(list_aaindex)/len(list_aaindex)
    elif len(list_aaindex) >= 3:
        list_aaindex.sort(reverse=True)
        list_aaindex = list_aaindex[:3]
        ave_aaindex = sum(list_aaindex)/len(list_aaindex)

    return ave_aaindex


# list_allindex = ['PONP800103', 'PONP800105', 'PONP800106', 'WILM950102', 'WILM950103', 'WILM950104', 'JURD980101', 'ENGD860101', 'FAUJ880109', 'FAUJ880108', 'ZIMJ680103', 'JANJ780101', 'RADA880106']
list_allindex = ['PONJ960101', 'ZIMJ680104', 'CHAM820101', 'CHOC760102']
df_BY = pd.read_csv(r'../../../../data/result/BYHA1_predictdata.csv',index_col=0)
for code in tqdm(list_allindex):
    aaindex_dict = aaindex1[code].values
    aaindex_dict.pop('-')
    print(aaindex_dict)
    name = 'x_' + code
    df_BY[name] = df_BY.parallel_apply(lambda row: AAindex(row['virus1_seq'], row['virus2_seq']), axis=1)
df_BY.to_csv(r'../../../../data/result/BYHA1_predictdata.csv')


