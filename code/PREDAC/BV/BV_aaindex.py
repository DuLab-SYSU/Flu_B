# -*- coding: utf-8 -*-
# @Time    : 2023/12/22 下午3:04
# @Author  : Hanwenjie
# @project : code
# @File    : BV_aaindex.py
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
    '''We will upload the complete code here once the manuscript is officially published'''

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

df_BV = pd.read_csv(r'../data/PREDAC/BV_model.csv',index_col=0)

for code in tqdm(list_allindex):
    aaindex_dict = aaindex1[code].values
    aaindex_dict.pop('-')
    print(aaindex_dict)
    name = 'x_' + code
    df_BV[name] = df_BV.parallel_apply(lambda row: AAindex(row['virus1_seq'], row['virus2_seq']), axis=1)
df_BV.to_csv(r'../data/BV_model.csv')

