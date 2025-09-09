# -*- coding: utf-8 -*-
# @Time    : 2023/12/24 下午2:55
# @Author  : Hanwenjie
# @project : code
# @File    : BY_aaindex.py
# @IDE     : PyCharm

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
    '''calculate the physical chemical properties of amino acids（aaindex）'''
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

#calculate all the physical chemical properties of 5 kinds
list_hydro = ['ARGP820101','CIDH920101','CIDH920102','CIDH920103','CIDH920104','CIDH920105','EISD840101','GOLD730101','JOND750101',
              'MANP780101','PONP800101','PONP800102','PONP800103','PONP800104','PONP800105','PONP800106',
              'PRAM900101','SWER830101','ZIMJ680101','PONP930101','WILM950101','WILM950102','WILM950103',
              'WILM950104','JURD980101','WOLR790101','KIDA850101','COWR900101','BLAS910101','CASG920101','ENGD860101','FASG890101','FAUJ880109']

list_volume = ['BIGC670101','BULH740102','CHOC750101','COHE430101','FAUJ880103','GOLD730102','GRAR740103','KRIW790103',
               'TSAJ990101','TSAJ990102','HARY940101','PONJ960101','FAUJ880108']

list_isoelectric = ['ZIMJ680104']
list_polarity = ['CHAM820101','GRAR740102','RADA880108','WOEC730101','ZIMJ680103']
list_surface = ['CHOC760101','CHOC760102','JANJ780101','RADA880106']

list_allindex = list_hydro + list_volume + list_isoelectric + list_polarity + list_surface

df_BY = pd.read_csv(r'../data/PREDAC/BY_model.csv',index_col=0)
for code in tqdm(list_allindex):
    aaindex_dict = aaindex1[code].values
    aaindex_dict.pop('-')
    print(aaindex_dict)
    name = 'x_' + code
    df_BY[name] = df_BY.parallel_apply(lambda row: AAindex(row['virus1_seq'], row['virus2_seq']), axis=1)
df_BY.to_csv(r'../result/BY_model.csv')


