# -*- coding: utf-8 -*-
# @Time    : 2024/1/8 下午2:38
# @Author  : Hanwenjie
# @project : code
# @File    : BVcross_aaindex_update.py
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


list_allindex = ['FAUJ880109', 'FAUJ880103', 'ZIMJ680104', 'ZIMJ680103', 'CHOC760101']
df_BV = pd.read_csv(r'../../../../data/result/BVHA1_predictdata_update.csv',index_col=0)
for code in tqdm(list_allindex):
    aaindex_dict = aaindex1[code].values
    aaindex_dict.pop('-')
    print(aaindex_dict)
    name = 'x_' + code
    df_BV[name] = df_BV.parallel_apply(lambda row: AAindex(row['virus1_seq'], row['virus2_seq']), axis=1)
df_BV.to_csv(r'../../../../data/result/BVHA1_predictdata_update.csv')

