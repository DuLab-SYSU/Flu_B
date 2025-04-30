#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scipy.stats import qmc, norm
import subprocess
from multiprocessing import Pool
import itertools
import time

# start time
start_time = time.time()

def run_script(params):
    param1, param2, param3, param4, param5, param6, param7, param8  = params
    command = ["python", "flu_b_ci.py", param1, param2, param3, param4, param5, param6, param7, param8]
    subprocess.run(command)

df_prior = pd.read_csv(r'../data/B_params_rank.csv', index_col=0)


list_params = ['w_v', 'w_y', 'cv', 'cy', 'tv_0', 'ty_0', 'r']

dict_params = {}
dict_params_sd = {}

# determine the mean value of params BV and BY
for param in list_params:
    param_mean = param + '_mean'
    dict_params[param] = df_prior.loc[0, param_mean]

# determine the sd of params BV and BY
for param in list_params:
    param_std = param + '_std'
    dict_params_sd[param] = df_prior.loc[0, param_std]

# set params
N_simulations = 1000 # num of samples
'''We will upload the complete code here once the manuscript is officially published'''

# tranfer float to str type
samples = param_samples.tolist()
samples = [[str(x) for x in row] for row in samples]

# add file number
for i, sublist in enumerate(samples):
    sublist.append(str(i+1))

# import prior samples
df = pd.DataFrame(samples)
df.columns = ['w_v', 'w_y', 'cv', 'cy', 
              'tv_0', 'ty_0', 'r', 'group']
df.to_csv(r"../result/dynamic_model/CI/prior_samples.csv", index=False)

if __name__ == "__main__":
    params_list = samples
    
    with Pool(processes=66) as pool: 
        pool.map(run_script, params_list)

# end time 
end_time = time.time()
# delta time
elapsed_time = end_time - start_time
# transfer second to hour
elapsed_hours = elapsed_time / 3600

with open(r"../result/dynamic_model/CI/time.txt", 'w+') as f:
    f.write(str(elapsed_hours) + 'hours')
