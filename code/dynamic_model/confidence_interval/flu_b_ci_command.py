#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import pandas as pd
from scipy.stats import qmc, norm
import subprocess
from multiprocessing import Pool
import itertools
import time


# In[29]:


# In[30]:


def run_script(params):
    param1, param2, param3, param4, param5, param6, param7, param8, param9, param10, param11  = params
    command = ["python", "flu_b_ci.py", param1, param2, param3, param4, param5, param6, param7, param8, param9, param10, param11]
    subprocess.run(command)


# In[31]:


df_prior = pd.read_csv(r'../../../data/result/dynamic_model/B_params_rank.csv', index_col=0)


# In[32]:


list_params = ['R0_v', 'R0_y', 'w_v', 'w_y', 'cv', 'cy', 'tv_0', 'ty_0', 'det_prob_v', 'det_prob_y']


# In[33]:


dict_params = {}
dict_params_sd = {}


# In[34]:


# determine the mean value of params BV and BY
for param in list_params:
    param_mean = param + '_mean'
    dict_params[param] = df_prior.loc[0, param_mean]


# In[35]:


dict_params


# In[36]:


# determine the sd of params BV and BY
for param in list_params:
    param_std = param + '_std'
    dict_params_sd[param] = df_prior.loc[0, param_std]


# In[37]:


dict_params_sd


# In[38]:


# 设定参数
N_simulations = 1000  # 采样次数

param_means = np.array([dict_params['R0_v'], dict_params['R0_y'],
                        dict_params['w_v'], dict_params['w_y'], dict_params['cv'], dict_params['cy'],
                        dict_params['tv_0'], dict_params['ty_0'], dict_params['det_prob_v'], dict_params['det_prob_y']])
param_stds = np.array([dict_params_sd['R0_v'], dict_params_sd['R0_y'],
                       dict_params_sd['w_v'], dict_params_sd['w_y'], dict_params_sd['cv'], dict_params_sd['cy'],
                       dict_params_sd['tv_0'], dict_params_sd['ty_0'], dict_params_sd['det_prob_v'], dict_params_sd['det_prob_y']])
num_params = len(param_means)  # 参数个数

# 1. 生成拉丁超立方样本（均匀分布在 [0,1]）
sampler = qmc.LatinHypercube(d=num_params)  # 7 维参数空间
lhs_samples = sampler.random(n=N_simulations)  # 生成 N_simulations 组样本

# 2. 将均匀分布转换为正态分布（逆变换法）
param_samples = norm.ppf(lhs_samples, loc=param_means, scale=param_stds)  # 逐维映射


# In[11]:


# tranfer float to str type
samples = param_samples.tolist()
samples = [[str(x) for x in row] for row in samples]


# In[12]:


# add file number
for i, sublist in enumerate(samples):
    sublist.append(str(i+1))


# In[16]:


len(samples)


# In[18]:


# import prior samples
# samples = samples[:10].copy()
df = pd.DataFrame(samples)
df.columns = ['R0_v', 'R0_y', 'w_v', 'w_y', 'cv', 'cy', 'tv_0', 'ty_0', 'det_prob_v', 'det_prob_y', 'group']
df.to_csv(r"../../../data/result/fluB_ci/prior_samples.csv", index=False)


# In[ ]:


if __name__ == "__main__":
    params_list = samples
    
    with Pool(processes=30) as pool: 
        pool.map(run_script, params_list)


# In[ ]:

