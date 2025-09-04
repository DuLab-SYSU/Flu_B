#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp


# In[2]:


# Function to process one set of parameters
def process_extinction_time(args):
    cv, cy = args
    df = pd.read_csv(f"./result/simmulation/data_c/data_simmulation_orign_cv{cv:.2f}_cy{cy:.2f}.csv", index_col='time_plot')
    df_r = df[df.index >= 2020].copy()

    bv_combi = f'BV_y_cv{cv:.2f}_cy{cy:.2f}'
    by_combi = f'BY_y_cv{cv:.2f}_cy{cy:.2f}'

    # calculate the eidemic size of 19/20 season
    bv_size = df[bv_combi].iloc[206:258].sum() 
    by_size = df[by_combi].iloc[206:258].sum()
    
    # Change values less than 1 to 0
    df_r[bv_combi] = df_r[bv_combi].where(df_r[bv_combi] >= 1, 0)
    df_r[by_combi] = df_r[by_combi].where(df_r[by_combi] >= 1, 0)

    # Select the extinction time by continuous 3 weeks of zero new cases
    bv_rolling = df_r[bv_combi].rolling(window=3, min_periods=3).sum().shift(-2)
    by_rolling = df_r[by_combi].rolling(window=3, min_periods=3).sum().shift(-2)

    bv_index = bv_rolling[bv_rolling == 0].index
    by_index = by_rolling[by_rolling == 0].index

    # Extinction combination indicators
    bv_combi_indicator = bv_combi
    by_combi_indicator = by_combi

    # Identify the extinction time
    bv_time = bv_index[0] if not bv_index.empty else 10000
    by_time = by_index[0] if not by_index.empty else 10000

    return bv_combi_indicator, bv_time, by_combi_indicator, by_time, bv_size, by_size


# In[3]:


# Define parameter ranges
cv_values = np.arange(0.0, 1.01, 0.01)
cy_values = np.arange(0.0, 1.01, 0.01)

# Generate all parameter combinations
param_list = [(cv, cy) for cv in cv_values for cy in cy_values]

# Use multiprocessing Pool
with mp.Pool(processes=160) as pool:
    results = list(tqdm(pool.imap(process_extinction_time, param_list), total=len(param_list)))

# Convert results into dictionaries
dict_bv_time = {bv_key: bv_val for bv_key, bv_val, _, _, _, _ in results}
dict_bv_size = {bv_key: bv_val for bv_key, _, _, _, bv_val, _ in results}
dict_by_time = {by_key: by_val for _, _, by_key, by_val, _, _ in results}
dict_by_size = {by_key: by_val for _, _, by_key, _, _, by_val in results}


# In[4]:


# 转换为 DataFrame
df_by_time = pd.DataFrame.from_dict(dict_by_time, orient='index', columns=['extinction_time'])
df_by_time.reset_index(inplace=True)
df_by_time.rename(columns={'index': 'combination'}, inplace=True)
df_by_time.to_csv(r"./result/simmulation/by_extinction_time_changeC.csv")

df_by_size = pd.DataFrame.from_dict(dict_by_size, orient='index', columns=['extinction_size'])
df_by_size.reset_index(inplace=True)
df_by_size.rename(columns={'index': 'combination'}, inplace=True)
df_by_size.to_csv(r"./result/simmulation/by_extinction_size_changeC.csv")

