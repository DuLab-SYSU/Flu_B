#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from pyDOE import lhs
import subprocess
from multiprocessing import Pool
import time

# start time
start_time = time.time()

def run_script(params):
    param1, param2, param3, param4, param5, param6, param7, param8, param9, param10, param11, param12, param13, param14 = params
    command = ["python", "flu_b_mcmc_parallel.py", param1, param2, param3, param4, param5, param6, param7, param8, param9, param10, param11, param12, param13, param14]
    subprocess.run(command)

# random seed
np.random.seed(42)

# prior distribution
params = {
    'w_v': [0.01, 0.99],
    'w_y': [0.01, 0.99],
    'cv': [0.01, 0.95],
    'cy': [0.01, 0.95],
    'tv_0': [12, 22.],
    'ty_0': [12, 22.],
    'r': [0.1, 0.95],
    'S_rate_ini': [0.1, 0.9],
    'Rcv_rate_ini': [0.01, 0.1],
    'Rcy_rate_ini': [0.01, 0.1],
    'Rv_rate_ini': [0.1, 0.9],
    'Ry_rate_ini': [0.1, 0.9],
    'R_rate_ini': [0.1, 0.9],
}

# map each each param to [0, 1] range
lower_bounds = np.array([param[0] for param in params.values()])
upper_bounds = np.array([param[1] for param in params.values()])

# lhs sampling
n_samples = 5000
lhs_samples = lhs(len(params), samples=n_samples)

# transfer sample range to params range
samples = lower_bounds + lhs_samples * (upper_bounds - lower_bounds)

# tranfer float to str type
samples = samples.tolist()
samples = [[str(x) for x in row] for row in samples]

# add file number
for i, sublist in enumerate(samples):
    sublist.append(str(i+1))

# import prior samples
df = pd.DataFrame(samples)
df.columns = ['w_v', 'w_y', 'cv', 'cy', 
              'tv_0', 'ty_0', 'r',
              'S_rate_ini', 'Rcv_rate_ini', 'Rcy_rate_ini', 
              'Rv_rate_ini', 'Ry_rate_ini', 'R_rate_ini', 
              'group']
df.to_csv(r"./result/dynamic_model/prior_samples.csv", index=False)

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

with open(r"./result/dynamic_model/time.txt", 'w+') as f:
    f.write(str(elapsed_hours) + 'hours')

