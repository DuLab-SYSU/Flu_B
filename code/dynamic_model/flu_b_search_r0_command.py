#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import subprocess
from multiprocessing import Pool
import itertools
import time

# start time
start_time = time.time()

def run_script(params):
    param1, param2, param3 = params
    command = ["python", "flu_b_search_r0.py", param1, param2, param3]
    subprocess.run(command)

# prior combination
R0_v_values = np.round(np.arange(1.050, 1.500, 0.001), 3)  # 生成并四舍五入到 3 位小数
R0_y_values = np.round(np.arange(1.050, 1.500, 0.001), 3)  

# generate all combinations
all_combinations = list(itertools.product(R0_v_values, R0_y_values))

# select R0_v >= R0_y + 0.05 combination
samples = [[str(R0_v), str(R0_y)] for R0_v, R0_y in all_combinations if R0_v >= R0_y + 0.05]

# add file number
for i, sublist in enumerate(samples):
    sublist.append(str(i+1))

# import prior samples
df = pd.DataFrame(samples)
df.columns = ['R0_v', 'R0_y', 'group']
df.to_csv(r"./result/fluB_r/prior_samples.csv", index=False)

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

with open(r"./result/dynamic_model/R0/time.txt", 'w+') as f:
    f.write(str(elapsed_hours) + 'hours')