# -*- coding: utf-8 -*-
# @Time    : 2023/12/25 9:43 PM
# @Author  : Hanwenjie
# @project : code
# @File    : BV_MCL_command.py
# @IDE     : PyCharm
# @REMARKS : description
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42
import pandas as pd
from tqdm import tqdm
import os
import networkx as nx
from MCL_function import mcl_modularity, vaccine_cluster, cat_cluster


# Step 1: Select time-based virus pairs from all predicted virus pairs -- after 1987 and remove abnormal strains
# Import the overall prediction file
df_all = pd.read_csv(r'./result/BVHA1_predictdata_update.csv',index_col=0)

# Import the strain number list to be used for MCL
df_num_sample = pd.read_csv(r'./result/BV_sample_final.csv',index_col=0)
list_num = list(df_num_sample['num'])

# Extract the above strain numbers from the prediction data
df_num_1987 = df_all.loc[(df_all['virus1_year'] >= 1987) & (df_all['virus2_year'] >= 1987)]
df_num_1987 = df_num_1987[df_num_1987['virus1_num'].isin(list_num)]
df_num_1987 = df_num_1987[df_num_1987['virus2_num'].isin(list_num)]
df_num_1987.reset_index(inplace=True,drop=True)
print(df_num_1987.shape[0])


# Step 2: Prepare data for MCL -- similarity network
df_similar = df_num_1987[df_num_1987['anti_ratio'] > 1].copy()
df_similar.reset_index(inplace=True,drop=True)
# print(df_similar.shape[0])
# print(df_similar.head())
print(df_num_1987['similarity'].value_counts())
# print(df_similar.columns)

df_MCL = df_similar.loc[:,['virus1_description','virus2_description','anti_ratio']]
df_MCL_log = df_similar.loc[:,['virus1_description','virus2_description','log_ratio']]
print(df_MCL.shape[0])
print(df_MCL_log.shape[0])
# print(df_MCL.head())

'''
# Step 3: Write the text files for MCL
# Write MCL file
print(df_MCL.shape[0])
with open(r'./result/BV_MCL.txt','w+') as f:
    for line in df_MCL.values:
        f.write((str(line[0]) + '\t' + str(line[1]) + '\t' + str(line[2]) + '\n'))

# Write log_MCL file
print(df_MCL_log.shape[0])
with open(r'./result/BV_MCL_log.txt','w+') as f:
    for line in df_MCL_log.values:
        f.write((str(line[0]) + '\t' + str(line[1]) + '\t' + str(line[2]) + '\n'))


# Step 4: Run MCL terminal command + plateau curve
# Open MCL txt data file path
open_file = "cd ./result/" \
            "MCL"
# Open log_MCL txt data file path
open_log = "cd ..; cd log_MCL"

# Path to MCL code
mcl_code = "./code/batch_mcl.py"

# MCL files
mcl_txt = "BV_MCL.txt"
mcl_log_txt = "BV_MCL_log.txt"

# MCL commands
com_mcl = "python " + mcl_code + " " + mcl_txt + " " + "1.0 150"
com_mcl_log = "python " + mcl_code + " " + mcl_log_txt + " " + "1.0 150"

# Plateau curve
com_inf = "python " + "./code/Mean_Cluster_Size.py lit2020"

# Combine commands
command = open_file + "; " + com_mcl + "; " + com_inf + "; " + open_log + "; " + com_mcl_log + "; " + com_inf

# Run system command
os.system(command)
'''

# Dataset path used later
path_data = "./result/MCL/" \
            "nosample_update"

# Step 5: Modularity curve plotting, calculate modularity, and draw the plot
modularity_MCL = mcl_modularity(path_in=path_data,df_net=df_MCL.copy(), kind_mcl='MCL')
modularity_logMCL = mcl_modularity(path_in=path_data,df_net=df_MCL_log.copy(), kind_mcl='log_MCL')

# Step 6: Check the antigenic cluster division of vaccine strains
vaccine_cluster(kind='MCL', path_max=modularity_MCL[0], infla=modularity_MCL[1])
vaccine_cluster(kind='log_MCL', path_max=modularity_logMCL[0], infla=modularity_logMCL[1])

# Step 7: Generate datasets for plotting MCL epidemic proportions
df_allseq = pd.read_csv(r"./result/BVHA1_usedfor_cluster.csv",index_col=0)

cat_cluster(kind='MCL',path_cluster=modularity_MCL[0], df_sample=df_num_sample.copy(), df_all=df_allseq.copy(), path_save=path_data)
cat_cluster(kind='log_MCL',path_cluster=modularity_logMCL[0], df_sample=df_num_sample.copy(), df_all=df_allseq.copy(), path_save=path_data)
