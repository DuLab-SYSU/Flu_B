# -*- coding: utf-8 -*-
# @Time    : 2023/12/25 9:44 PM
# @Author  : Hanwenjie
# @project : code
# @File    : MCL_function.py
# @IDE     : PyCharm
# @REMARKS : description text

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

# Step 5: Modularity curve plotting
dict_ratio = {'MCL': 'anti_ratio', 'log_MCL': 'log_ratio'}  # Specify log or non-log and the corresponding ratio


def mcl_modularity(path_in, df_net, kind_mcl):
    # Build the network based on imported data
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
    print(nodes_number)
    print(edges_number)

    # Calculate network modularity
    list_inflation = []  # Store inflation parameters for plotting
    list_modularity = []  # Store modularity values for each inflation parameter
    for j in tqdm(range(10, 151)):
        name_cluster = {}  # description-cluster dictionary
        # Data reading path
        path = path_in + "/" + kind_mcl + "/" + "batch_MCL_out/out.lit2020.I" + str(j)
        # print(path)
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

    # Output the best clustering file path and inflation parameter
    path_max = path_in + "/" + kind_mcl + "/" + "batch_MCL_out/out.lit2020.I" + \
               str(int(list_inflation[list_modularity.index(max(list_modularity))] * 10))
    infla = str(list_inflation[list_modularity.index(max(list_modularity))])

    # Plot modularity curve
    fig, ax = plt.subplots(layout='constrained', figsize=(12, 8))
    ax.plot(list_inflation, list_modularity, linewidth=4, clip_on=False)
    ax.scatter(list_inflation, list_modularity, s=120, clip_on=False)

    ax.set_ylim(0, 1)
    ax.set_yticks([i / 10 for i in range(1, 11)])

    ax.set_xlim(0.8, 15)
    ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])

    ax.text(list_inflation[list_modularity.index(max(list_modularity))], max(list_modularity),
            (list_inflation[list_modularity.index(max(list_modularity))], round(max(list_modularity), 3)))
    ax.axvline(x=list_inflation[list_modularity.index(max(list_modularity))], zorder=0, color='#bf3a2b', ls='--',
               linewidth='3')
    ax.axhline(y=max(list_modularity), zorder=0, color='#bf3a2b', ls='--', linewidth='3')
    plt.tight_layout()

    fig_path = path_in + "/" + kind_mcl + "/modularity.png"  # Save figure path
    # print(fig_path)

    plt.savefig(fig_path)
    plt.show()

    return path_max, infla


# Define a function to check vaccine strain clustering
def vaccine_cluster(kind, path_max, infla):
    print(kind + " vaccine strain clustering " + "(inflation parameter = " + infla + "):")
    s = 0
    with open(path_max, 'r') as f:
        lines = f.readlines()
        for line in lines:
            s += 1

            if 'EPI_ISL_95 | B/Victoria/2/87 | 1987-01-01 |  | B / H0N0 | Victoria | EPI130089 | Oceania' in str(
                    line):
                print('B/Victoria/2/1987:' + str(s))
            if 'EPI_ISL_2460 | B/Beijing/1/87 | 1987-01-01 |  | B / H0N0 | Victoria | EPI131390 | Asia' in str(
                    line):
                print('B/Beijing/1/1987:' + str(s))
            if 'EPI_ISL_2821 | B/Shandong/7/97 | 1997-01-01 |  | B / H0N0 | Victoria | EPI13573 | Asia' in str(
                    line):
                print('B/Shandong/7/1997:' + str(s))
            if 'EPI_ISL_2342 | B/Hong Kong/330/2001 | 2001-01-01 |  | B / H0N0 | Victoria | EPI20103 | Asia' in str(
                    line):
                print('B/HongKong/330/2001:' + str(s))
            if 'EPI_ISL_30253 | B/Malaysia/2506/2004 | 2004-01-01 | Egg 2 | B / H0N0 | Victoria | EPI180005 | Asia' in str(
                    line):
                print('B/Malaysia/2506/2004:' + str(s))
            if 'EPI_ISL_28587 | B/Brisbane/60/2008 | 2008-08-15 | E4 | B / H0N0 | Victoria | EPI173277 | Oceania' in str(
                    line):
                print('B/Brisbane/60/2008:' + str(s))
            if 'EPI_ISL_257735 | B/Colorado/06/2017 | 2017-02-25 | Original | B / H0N0 | Victoria | EPI969380 | NorthAmerica' in str(
                    line):
                print('B/Colorado/06/2017:' + str(s))
            if 'EPI_ISL_7473160 | B/Washington/02/2019    (21/336) | 2019-01-19 | E3/E2 | B / H0N0 | Victoria | EPI1942265 | NorthAmerica' in str(
                    line):
                print('B/Washington/02/2019:' + str(s))
            if 'EPI_ISL_16967998 | B/Austria/1359417/2021 | 2021-01-09 | S1C4/C2 (2022-07-31) | B / H0N0 | Victoria | EPI2413510 | Europe' in str(
                    line):
                print('B/Austria/1359417/2021:' + str(s))
        print('\n')


# Define a function to generate datasets for MCL prevalence plotting
# Change 3 parts: MCL file parameter, sampled data year, save file path
def cat_cluster(kind, path_cluster, df_sample, df_all, path_save):
    name_cluster = {}  # description-cluster dictionary
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

    name_seq = df_sample[['description', 'seq']].set_index('seq').to_dict()['description']  # seq-name dictionary

    # Match all sequences with clusters, excluding sequences not sampled
    e = 0
    for c in range(df_all.shape[0]):
        try:
            df_all.loc[c, 'cluster'] = name_cluster[name_seq[df_all.loc[c, 'seq']]]
        except KeyError:
            e += 1

    print("Total number of sequences:", df_all.shape[0])
    print("Number of unclassified sequences:", e)
    df_all.dropna(subset=['cluster'], axis=0, inplace=True)
    df_all.reset_index(inplace=True, drop=True)
    path_tosave = path_save + '/' + kind + '/' + 'BV_cluster.csv'
    df_all.to_csv(path_tosave)

    # Count the number and proportion of sequences for different epidemic seasons, used for plotting
    # Specify hemispheres and regions
    df_North = df_all[
        (df_all['region'] == 'NorthAmerica') | (df_all['region'] == 'Asia') | (df_all['region'] == 'Europe')]
    df_South = df_all[
        (df_all['region'] == 'SouthAmerica') | (df_all['region'] == 'Africa') | (df_all['region'] == 'Oceania')]
    df_NorthAmerica = df_all[(df_all['region'] == 'NorthAmerica')]
    df_Asia = df_all[(df_all['region'] == 'Asia')]
    df_Europe = df_all[(df_all['region'] == 'Europe')]
    df_SouthAmerica = df_all[(df_all['region'] == 'SouthAmerica')]
    df_Africa = df_all[(df_all['region'] == 'Africa')]
    df_Oceania = df_all[(df_all['region'] == 'Oceania')]

    # List of dataframes
    list_df = [df_North, df_South, df_NorthAmerica, df_Asia, df_Europe, df_SouthAmerica, df_Africa, df_Oceania]

    # Loop through and output cluster files for each hemisphere and region
    area_name = ['North', 'South', 'NorthAmerica', 'Asia', 'Europe', 'SouthAmerica', 'Africa', 'Oceania']
    area_num = 0  # Used for naming output files
    for df_area in list_df:
        df_area.reset_index(inplace=True, drop=True)

        df_cluster = pd.DataFrame()
        i = 0
        for group in df_area[['epidemic_season', 'cluster']].groupby(['epidemic_season']):

            df_cluster.loc[i, 'epidemic_season'] = group[0][0]
            # Count the proportion of each antigenic cluster per epidemic season
            for j in range(1, num_cluster + 1):
                cluster_num = 'cluster' + str(j) + '_' + 'num'
                cluster_per = 'cluster' + str(j) + '_' + 'proportion'
                try:
                    df_cluster.loc[i, cluster_num] = group[1].value_counts()[(group[0][0], j)]
                    df_cluster.loc[i, cluster_per] = group[1].value_counts(normalize=True)[(group[0][0], j)]
                    # print(i)
                except KeyError:
                    # print(i)
                    pass
            i += 1

        cols = list(df_cluster.columns)

        cluster_proportion = [cluster for cluster in cols if 'proportion' in cluster]
        df_proportion = df_cluster[cluster_proportion]

        # Determine predominant strains
        for i in range(df_proportion.shape[0]):
            row = df_proportion.loc[i, :]
            cluster_predominant = row[row == row.max()].index
            df_cluster.loc[i, 'cluster_predominant'] = np.random.choice(cluster_predominant)[:-11]

        # Count the number of predominant strain sequences
        cluster_num = [cluster for cluster in cols if 'num' in cluster]
        df_num = df_cluster[cluster_num]

        for i in range(df_num.shape[0]):
            row = df_num.loc[i, :]
            df_cluster.loc[i, 'num_predominant'] = row.max()

        # Write year and season
        df_cluster['year'] = df_cluster['epidemic_season'].str[:4]
        df_cluster['season'] = df_cluster['epidemic_season'].str[5:]

        df_cluster['year'] = df_cluster['year'].astype(int)
        df_cluster.fillna(0, inplace=True)

        df_winter = df_cluster[df_cluster['season'] == 'winter']
        df_summer = df_cluster[df_cluster['season'] == 'summer']
        df_winter.reset_index(inplace=True, drop=True)
        df_summer.reset_index(inplace=True, drop=True)

        # Count predominant antigenic clusters for each region
        cols = list(df_winter.columns)
        clusters = [cluster for cluster in cols if 'proportion' in cluster]
        clusters_copy = clusters.copy()
        for cluster in clusters_copy:
            if df_winter[cluster].max() < 0.1:
                clusters.remove(cluster)
        print(area_name[area_num] + " region has %s significant predominant antigenic clusters, which are: " % (str(len(set(df_cluster['cluster_predominant'])))), set(df_cluster['cluster_predominant']))

        path_areasave = path_save + '/' + kind + '/' + 'cluster_percentage' + '/' + 'BV' \
                        + area_name[area_num] + '_cluster_percentage.csv'
        df_cluster.to_csv(path_areasave)

        area_num += 1
