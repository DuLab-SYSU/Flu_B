# -*- coding: utf-8 -*-
# @Time    : 2023/12/22 上午11:01
# @Author  : Hanwenjie
# @project : code
# @File    : BV_match_seq_HI.py
# @IDE     : PyCharm
# @REMARKS : 说明文字
import pandas as pd

seq = pd.read_csv(r"../../../../sequence/BVHA1_usedfor_cluster.csv",index_col=0)
seq.rename(columns={'virus':'virus_1'},inplace=True)
HI = pd.read_csv(r'../../../../data/'
                 r'HI/BVHI.csv',index_col=0)

df_merge = pd.merge(HI,seq,on='virus_1')
df_merge.rename(columns={'seq_selfill':'virus1_seq','description':'virus1_description','ID':'virus1_ID','region':'virus1_region',
                         'collection_date':'virus1_collection_date','collection_year':'virus1_collection_year',
                         'collection_month':'virus1_collection_month'},inplace=True)
print(df_merge.shape[0])

seq.rename(columns={'virus_1':'virus_2'},inplace=True)
df_merge1 = pd.merge(df_merge,seq,on='virus_2')
df_merge1.rename(columns={'seq_selfill':'virus2_seq','description':'virus2_description','ID':'virus2_ID','region':'virus2_region',
                          'collection_date':'virus2_collection_date','collection_year':'virus2_collection_year',
                          'collection_month':'virus2_collection_month'},inplace=True)
print(df_merge1.shape[0])
df_final = df_merge1.sort_values(by='count',ascending=False)
df_final.reset_index(inplace=True,drop=True)
print(df_final.columns)

df_final.drop_duplicates(subset=['virus1_seq','virus2_seq'],keep='first',inplace=True)
df_final.drop(df_final.index[(df_final['virus1_seq'] == df_final['virus2_seq'])],inplace=True)
df_final.reset_index(drop=True,inplace=True)
print(df_final.shape[0])

df_final.to_csv(r'../../../../data/result/BV_model.csv')
