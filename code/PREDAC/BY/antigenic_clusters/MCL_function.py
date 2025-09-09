# -*- coding: utf-8 -*-
# @Time    : 2023/12/25 下午10:40
# @Author  : Hanwenjie
# @project : code
# @File    : MCL_function.py
# @IDE     : PyCharm
# @REMARKS : Functions for MCL modularity and cluster analysis

import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
import pandas as pd
import markov_clustering as mc
import networkx as nx
import random
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np

# Step 5: Plot modularity curve
dict_ratio = {'MCL': 'anti_ratio', 'log_MCL': 'log_ratio'}  # Specify log or non-log and corresponding ratio


def mcl_modularity(path_in, df_net, kind_mcl):
    # Construct network from imported data
    virus1_list = df_net['virus1_description'].values.tolist()
    virus2_list = df_net['virus2_description'].values.tolist()
    edge_list = df_net[dict_ratio[kind_mcl]].values.tolist()  # Determine ratio
    virus_net = nx.Graph()
    virus_net.add_nodes_from(virus1_list)
    virus_net.add_nodes_from(virus2_list)
    for x, y, z in zip(virus1_list, virus2_list, edge_list):
        virus_net.add_edge(x, y, weight=z)
    nodes_number = virus_net.number_of_nodes()
    edges_number = virus_net.number_of_edges()
    print("Number of nodes:", nodes_number)
    print("Number of edges:", edges_number)

    # Compute modularity
    list_inflation = []  # Store inflation values for plotting
    list_modularity = []  # Store modularity values for each inflation
    for j in tqdm(range(10, 121)):
        name_cluster = {}  # description-to-cluster dictionary
        path = path_in + "/" + kind_mcl + "/" + "batch_MCL_out/out.lit2020.I" + str(j)
        i = 0
        num = 0
        list_cluster = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                i += 1
                line = line.replace('\n', '')
                list_line = line.split(sep='\t')
                set_line = set(list_line)
                list_cluster.append(set_line)

        net_modularity = nx.community.modularity(virus_net, list_cluster)
        list_inflation.append(j / 10)
        list_modularity.append(net_modularity)

    # Output path of best clustering file and inflation parameter
    max_idx = list_modularity.index(max(list_modularity))
    path_max = path_in + "/" + kind_mcl + "/" + "batch_MCL_out/out.lit2020.I" + str(int(list_inflation[max_idx] * 10))
    infla = str(list_inflation[max_idx])

    # Plot modularity
    fig, ax = plt.subplots(layout='constrained', figsize=(12, 8))
    ax.plot(list_inflation, list_modularity, linewidth=4, clip_on=False)
    ax.scatter(list_inflation, list_modularity, s=120, clip_on=False)

    ax.set_ylim(0, 1)
    ax.set_yticks([i / 10 for i in range(1, 11)])
    ax.set_xlim(0.8, 15)
    ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])

    ax.text(list_inflation[max_idx], max(list_modularity),
            (list_inflation[max_idx], round(max(list_modularity), 3)))
    ax.axvline(x=list_inflation[max_idx], zorder=0, color='#bf3a2b', ls='--', linewidth=3)
    ax.axhline(y=max(list_modularity), zorder=0, color='#bf3a2b', ls='--', linewidth=3)
    plt.tight_layout()

    fig_path = path_in + "/" + kind_mcl + "/modularity.png"  # Path to save figure
    plt.savefig(fig_path)
    plt.show()

    return path_max, infla


# Function to inspect vaccine cluster classification
def vaccine_cluster(kind, path_max, infla):
    print(kind + " vaccine cluster classification (inflation parameter: " + infla + "):")
    s = 0
    with open(path_max, 'r') as f:
        lines = f.readlines()
        for line in lines:
            s += 1
            if 'EPI_ISL_20958 | B/Yamagata/16/88' in str(line):
                print('B/Yamagata/16/1988 at line:' + str(s))
            if 'EPI_ISL_20945 | B/Panama/45/90' in str(line):
                print('B/Panama/45/1990 at line:' + str(s))
            if 'EPI_ISL_969 | B/Beijing/184/93' in str(line):
                print('B/Beijing/184/1993 at line:' + str(s))
            if 'EPI_ISL_21113 | B/Sichuan/379/99' in str(line):
                print('B/Sichuan/379/1999 at line:' + str(s))
            if 'EPI_ISL_6949688 | B/Shanghai/361/2002' in str(line):
                print('B/Shanghai/361/2002 at line:' + str(s))
            if 'EPI_ISL_22470 | B/Belem/621/2005' in str(line):
                print('B/Belem/621/2005 at line:' + str(s))
            if 'EPI_ISL_22808 | B/Florida/4/2006' in str(line):
                print('B/Florida/4/2006 at line:' + str(s))
            if 'EPI_ISL_76921 | B/Wisconsin/01/2010' in str(line):
                print('B/Wisconsin/01/2010 at line:' + str(s))
            if 'EPI_ISL_138284 | B/Massachusetts/02/2012' in str(line):
                print('B/Massachusetts/02/2012 at line:' + str(s))
            if 'EPI_ISL_165881 | B/Phuket/3073/2013' in str(line):
                print('B/Phuket/3073/2013 at line:' + str(s))
        print('\n')


# Function to generate dataset for plotting MCL cluster proportions
def cat_cluster(kind, path_cluster, df_sample, df_all, path_save):
    name_cluster = {}  # description-to-cluster dictionary
    path = path_cluster

    l = 0
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            l += 1
            line = line.replace('\n', '')
            list_line = line.split('\t')
            for virus in list_line:
                name_cluster[virus] = l
    num_cluster = l  # number of antigenic clusters

    name_seq = df_sample[['description', 'seq']].set_index('seq').to_dict()['description']  # seq-to-name dict

    # Match all sequences to clusters (excluding sequences not sampled)
    e = 0
    for c in range(df_all.shape[0]):
        try:
            df_all.loc[c, 'cluster'] = name_cluster[name_seq[df_all.loc[c, 'seq']]]
        except KeyError:
            e += 1

    print("Total sequences:", df_all.shape[0])
    print("Unclassified sequences:", e)
    df_all.dropna(subset=['cluster'], axis=0, inplace=True)
    df_all.reset_index(inplace=True, drop=True)
    path_tosave = path_save + '/' + kind + '/' + 'BY_cluster.csv'
    df_all.to_csv(path_tosave)

    # Count sequences by epidemic season and region
    df_North = df_all[df_all['region'].isin(['NorthAmerica', 'Asia', 'Europe'])]
    df_South = df_all[df_all['region'].isin(['SouthAmerica', 'Africa', 'Oceania'])]
    df_NorthAmerica = df_all[df_all['region'] == 'NorthAmerica']
    df_Asia = df_all[df_all['region'] == 'Asia']
    df_Europe = df_all[df_all['region'] == 'Europe']
    df_SouthAmerica = df_all[df_all['region'] == 'SouthAmerica']
    df_Africa = df_all[df_all['region'] == 'Africa']
    df_Oceania = df_all[df_all['region'] == 'Oceania']

    list_df = [df_North, df_South, df_NorthAmerica, df_Asia, df_Europe, df_SouthAmerica, df_Africa, df_Oceania]
    area_name = ['North', 'South', 'NorthAmerica', 'Asia', 'Europe', 'SouthAmerica', 'Africa', 'Oceania']

    # Process each area and output cluster files
    for area_num, df_area in enumerate(list_df):
        df_area.reset_index(inplace=True, drop=True)

        df_cluster = pd.DataFrame()
        i = 0
        for group in df_area[['epidemic_season', 'cluster']].groupby(['epidemic_season']):
            df_cluster.loc[i, 'epidemic_season'] = group[0][0]
            for j in range(1, num_cluster + 1):
                cluster_num = 'cluster' + str(j) + '_num'
                cluster_per = 'cluster' + str(j) + '_proportion'
                try:
                    df_cluster.loc[i, cluster_num] = group[1].value_counts()[(group[0][0], j)]
                    df_cluster.loc[i, cluster_per] = group[1].value_counts(normalize=True)[(group[0][0], j)]
                except KeyError:
                    pass
            i += 1

        cols = list(df_cluster.columns)
        cluster_proportion = [cluster for cluster in cols if 'proportion' in cluster]
        df_proportion = df_cluster[cluster_proportion]

        # Determine predominant cluster
        for i in range(df_proportion.shape[0]):
            row = df_proportion.loc[i, :]
            cluster_predominant = row[row == row.max()].index
            df_cluster.loc[i, 'cluster_predominant'] = np.random.choice(cluster_predominant)[:-11]

        # Count sequences of predominant clusters
        cluster_num = [cluster for cluster in cols if 'num' in cluster]
        df_num = df_cluster[cluster_num]
        for i in range(df_num.shape[0]):
            row = df_num.loc[i, :]
            df_cluster.loc[i, 'num_predominant'] = row.max()

        # Add year and season columns
        df_cluster['year'] = df_cluster['epidemic_season'].str[:4].astype(int)
        df_cluster['season'] = df_cluster['epidemic_season'].str[5:]
        df_cluster.fillna(0, inplace=True)

        df_winter = df_cluster[df_cluster['season'] == 'winter'].reset_index(drop=True)
        df_summer = df_cluster[df_cluster['season'] == 'summer'].reset_index(drop=True)

        # Identify significant clusters in winter
        cols = list(df_winter.columns)
        clusters = [cluster for cluster in cols if 'proportion' in cluster]
        clusters_copy = clusters.copy()
        for cluster in clusters_copy:
            if df_winter[cluster].max() < 0.1:
                clusters.remove(cluster)
        print(area_name[area_num] + " area significant clusters: %s, clusters:" % len(set(df_cluster['cluster_predominant'])), set(df_cluster['cluster_predominant']))

        path_areasave = path_save + '/' + kind + '/cluster_percentage/' + 'BY' + area_name[area_num] + '_cluster_percentage.csv'
        df_cluster.to_csv(path_areasave)

        area_num += 1

