# -*- coding: utf-8 -*-
# @Time    : 2023/12/25 22:40
# @Author  : Hanwenjie
# @project : code
# @File    : BY_MCL_command.py
# @IDE     : PyCharm
# @REMARKS : Run MCL analysis and generate modularity/cluster datasets

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
import pandas as pd
from tqdm import tqdm
import os
import networkx as nx
from MCL_function import mcl_modularity, vaccine_cluster, cat_cluster

# Step 1: Filter virus pairs based on year (after 1988) and remove outliers
df_all = pd.read_csv(
    r'./result/BYHA1_predictdata.csv', index_col=0
)

# Import virus sample list for MCL
df_num_sample = pd.read_csv(
    r'./result/BY_sample_dropoutlier.csv',
    index_col=0
)
list_num = list(df_num_sample['num'])

# Select relevant virus pairs from prediction data
df_num_1988 = df_all.loc[(df_all['virus1_year'] >= 1988) & (df_all['virus2_year'] >= 1988)]
df_num_1988 = df_num_1988[df_num_1988['virus1_num'].isin(list_num)]
df_num_1988 = df_num_1988[df_num_1988['virus2_num'].isin(list_num)]
df_num_1988.reset_index(inplace=True, drop=True)
print("Filtered virus pairs count:", df_num_1988.shape[0])

# Step 2: Prepare similarity network for MCL
df_similar = df_num_1988[df_num_1988['anti_ratio'] > 1].copy()
df_similar.reset_index(inplace=True, drop=True)
print("Similarity counts:\n", df_num_1988['similarity'].value_counts())

df_MCL = df_similar[['virus1_description', 'virus2_description', 'anti_ratio']]
df_MCL_log = df_similar[['virus1_description', 'virus2_description', 'log_ratio']]
print("MCL edges:", df_MCL.shape[0])
print("Log-MCL edges:", df_MCL_log.shape[0])

'''
# Step 3: Write MCL input files
with open(r'./result/MCL/BY_MCL.txt','w+') as f:
    for line in df_MCL.values:
        f.write(f"{line[0]}\t{line[1]}\t{line[2]}\n")

with open(r'./result/log_MCL/BY_MCL_log.txt','w+') as f:
    for line in df_MCL_log.values:
        f.write(f"{line[0]}\t{line[1]}\t{line[2]}\n")
'''

'''
# Step 4: Run MCL commands and mean cluster size calculation (optional)
open_file = "cd ./result/MCL/nosample_new/MCL"
open_log = "cd ..; cd log_MCL"
mcl_code = "./code/batch_mcl.py"
mcl_txt = "BY_MCL.txt"
mcl_log_txt = "BY_MCL_log.txt"
com_mcl = f"python {mcl_code} {mcl_txt} 1.0 150"
com_mcl_log = f"python {mcl_code} {mcl_log_txt} 1.0 150"
com_inf = "python ./code/Mean_Cluster_Size.py lit2020"
command = f"{open_file}; {com_mcl}; {com_inf}; {open_log}; {com_mcl_log}; {com_inf}"
os.system(command)
'''

# Path to processed MCL data
path_data = "./result/MCL/nosample_new"

# Step 5: Compute modularity curves and plot
modularity_MCL = mcl_modularity(path_in=path_data, df_net=df_MCL.copy(), kind_mcl='MCL')
modularity_logMCL = mcl_modularity(path_in=path_data, df_net=df_MCL_log.copy(), kind_mcl='log_MCL')

# Step 6: Inspect vaccine strain cluster assignment
vaccine_cluster(kind='MCL', path_max=modularity_MCL[0], infla=modularity_MCL[1])
vaccine_cluster(kind='log_MCL', path_max=modularity_logMCL[0], infla=modularity_logMCL[1])

# Step 7: Generate dataset for plotting MCL cluster proportions
df_allseq = pd.read_csv(
    r"./result/BYHA1_usedfor_cluster.csv",
    index_col=0
)

cat_cluster(kind='MCL', path_cluster=modularity_MCL[0], df_sample=df_num_sample.copy(), df_all=df_allseq.copy(), path_save=path_data)
cat_cluster(kind='log_MCL', path_cluster=modularity_logMCL[0], df_sample=df_num_sample.copy(), df_all=df_allseq.copy(), path_save=path_data)
