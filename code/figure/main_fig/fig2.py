# -*- coding: utf-8 -*-
# @Time    : 2025/4/1 下午2:26
# @Author  : Hanwenjie
# @project : BY_dn_ds_cluster_MEME.py
# @File    : fig2.py
# @IDE     : PyCharm
# @REMARKS : description text
import os
import math
import baltic as bt
import dendropy
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scienceplots
from scipy import stats
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.transforms import ScaledTranslation

mpl.rcParams['pdf.fonttype'] = 42

# import model performance data
df_BV = pd.read_csv(
    r"./result/BV_model_performance.csv", index_col=0)
df_BY = pd.read_csv(
    r"./result/BY_model_performance.csv", index_col=0)
# specify unit conversion to cm
cm = 2.54

# plotting
fig = plt.figure(figsize=(18.4 / cm, 12 / cm), dpi=300, layout="compressed")
spec = fig.add_gridspec(6, 3, width_ratios=[1, 1, 1.35])
trans = ScaledTranslation(-30 / 300, 30 / 300, fig.dpi_scale_trans)
with mpl.rc_context(
        {'font.family': 'Arial', 'font.size': 7, 'hatch.linewidth': .5, 'lines.linewidth': .3, 'patch.linewidth': .3,
         'xtick.labelsize': 6, 'ytick.labelsize': 6}):
    # A panel
    ax = fig.add_subplot(spec[0:2, 0])
    ax.text(-0.2, 0.97, 'A', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    # BV cross-validation evalutation
    metrics = ('AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score')
    metrics_value = [df_BV.query("model == 'lightgbm'")['auc_cv'].values[0],
                     df_BV.query("model == 'lightgbm'")['accuracy_cv'].values[0],
                     df_BV.query("model == 'lightgbm'")['precision_cv'].values[0],
                     df_BV.query("model == 'lightgbm'")['recall_cv'].values[0],
                     df_BV.query("model == 'lightgbm'")['f1-score_cv'].values[0]]

    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # the width of the bars

    ax.bar(x, metrics_value, width, color='#62B197', alpha=0.95, zorder=5)
    ax.tick_params(which='major', direction='out', bottom=False)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(axis='x', pad=0)
    ax.tick_params(axis='y', pad=1)
    # ax.set_title('B/Victoria',pad=10,)
    ax.grid(axis='y', ls='--', zorder=0, alpha=0.5)
    ax.spines['bottom'].set_zorder(10)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Performance')
    # ax.set_axisbelow(True)

    # B panel
    ax = fig.add_subplot(spec[0:2, 1])
    ax.text(-0.2, 0.97, 'B', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    # BY cross-validation evalutation
    metrics = ('AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score')
    metrics_value = [df_BY.query("model == 'xgboost'")['auc_cv'].values[0],
                     df_BY.query("model == 'xgboost'")['accuracy_cv'].values[0],
                     df_BY.query("model == 'xgboost'")['precision_cv'].values[0],
                     df_BY.query("model == 'xgboost'")['recall_cv'].values[0],
                     df_BY.query("model == 'xgboost'")['f1-score_cv'].values[0]]

    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # the width of the bars

    ax.bar(x, metrics_value, width, color='#E18E6D', alpha=0.95, zorder=5)
    ax.tick_params(which='major', direction='out', bottom=False)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(axis='x', pad=0)
    ax.tick_params(axis='y', pad=1)
    # ax.set_title('B/Victoria',pad=10,)
    ax.grid(axis='y', ls='--', zorder=0, alpha=0.5)
    ax.spines['bottom'].set_zorder(10)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Performance')
    # ax.set_axisbelow(True)

    # C panel
    ax = fig.add_subplot(spec[2:6, 0])
    ax.text(-0.2, 0.94, 'C', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    features = ['EpitopeA', 'EpitopeB', 'EpitopeC', 'EpitopeD', 'EpitopeE',
                'Property1', 'Property2', 'Property3', 'Property4', 'Property5',
                'Nglycosylation', 'Receptor binding']
    features_newname = {'EpitopeA': 'BV_E', 'EpitopeB': 'BV_A', 'EpitopeC': 'BV_D', 'EpitopeD': 'BV_C',
                        'EpitopeE': 'BV_B',
                        'Property1': 'Hydrophobicity', 'Property2': 'Volume', 'Property3': 'Charge',
                        'Property4': 'Polarity',
                        'Property5': 'ASA', 'Nglycosylation': 'N-Gly', 'Receptor binding': 'RBS'}

    contributions = np.array([0.313, 3.60387486, 0.55467476, 1.90445524, 0.65409724, 0.85292539,
                              0.59821007, 0.76302093, 1.74082968, 0.56377183, 0.64845957, 1.02104887])

    contribution_per = [round((contribution / np.sum(contributions)) * 100, 2) for contribution in contributions]
    fea_con = dict(zip(contribution_per, features))
    con_sort = sorted(contribution_per, reverse=False)
    fea_sort = [features_newname[fea_con[con]] for con in con_sort]
    fea_num = list(np.arange(1, 13, 1))
    feature_size = [i * 5 for i in con_sort]

    # colors = ['#839DD1', '#839DD1', '#839DD1', '#839DD1', '#839DD1',
    #           '#FBCF4B', '#FBCF4B', '#FBCF4B', '#FBCF4B', '#FBCF4B',
    #           '#F69877',
    #           '#A2D2BF']
    colors = ['#A2D2BF', '#A2D2BF', '#A2D2BF', '#A2D2BF', '#A2D2BF',
              '#F69877', '#F69877', '#F69877', '#F69877', '#F69877',
              '#839DD1',
              '#FBCF4B']
    cor_con = dict(zip(contribution_per, colors))
    cor_sort = [cor_con[con] for con in con_sort]

    # Mark the contribution of each feature
    for i in range(len(fea_sort) - 1):
        ax.text(x=con_sort[i] + 1.1, y=fea_num[i] - 0.15, s=str(con_sort[i]) + '%', fontsize=6)

    # Mark the last one separately; otherwise, it will go beyond the frame
    BV_lastnum = len(fea_sort) - 1
    ax.text(x=con_sort[BV_lastnum] - 7.5, y=fea_num[BV_lastnum] - 0.15, s=str(con_sort[BV_lastnum]) + '%', fontsize=6)
    # Set the detail format
    ax.scatter(con_sort, fea_num, s=feature_size, c=cor_sort, linewidths=0.5, edgecolors='grey', alpha=0.8, zorder=100)
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xticklabels(['0', '5%', '10%', '15%', '20%', '25%', '30%'])
    ax.set_xlabel('Contribution')
    ax.set_ylim(0.5, 12.5)
    ax.set_yticks(np.arange(1, 13, 1))
    ax.set_yticklabels(fea_sort)
    ax.tick_params(axis='x', labelsize=6, pad=1)
    ax.tick_params(axis='y', labelsize=6, pad=1)
    # ax.set_title('B/Victoria', fontsize=10)
    ax.grid(ls=':', zorder=0, alpha=0.5)
    ax.set_axisbelow(True)
    # ax.set_ylabel('Features')

    size = 0.55
    con_group = [np.sum(contribution_per[:5]), np.sum(contribution_per[5:8]), np.sum(contribution_per[8]),
                 np.sum(contribution_per[9])]
    # color = ['#82B0D2', '#FFBE7A', '#8ECFC9', '#FA7F6F']
    color = ['#A2D2BF', '#F69877', '#839DD1', '#FBCF4B']
    BV_labels = ['Epitopes', 'Physicochemical \n Properties', 'Glycosylation \n sites', 'Receptor \n Binding Sites']
    # Nested scatter plot
    pie_BV = fig.add_axes([0.16, 0.12, 0.16, 0.16])
    wedges, texts, autotexts = pie_BV.pie(con_group, radius=1.3, colors=color, autopct='%1.2f%%', shadow=False,
                                          pctdistance=0.81,
                                          labels=BV_labels, wedgeprops=dict(width=size, edgecolor='w'),
                                          textprops={"weight": "bold", "multialignment": "center"})

    # Adjust the font size
    for i, (text, autotext, wedge) in enumerate(zip(texts, autotexts, wedges)):
        size = con_group[i]
        proportion = size / sum(con_group)
        fontsize = 8 + 5 * proportion 
        # text.set_fontsize(fontsize)
        autotext.set_fontsize(fontsize)
        autotext.set_color("w")
        autotext.set_text(f'{proportion:.1%}') 

    # Adjust the percentage Angle and position
    autotexts[0].set_text("60.8%")
    autotexts[0].set_position((0.1, 0.92))
    autotexts[0].set_rotation(0)
    autotexts[1].set_rotation(-20)
    autotexts[2].set_rotation(35)
    autotexts[3].set_fontsize(6)
    autotexts[3].set_rotation(0)

    # Adjust the Angle and position of the label
    texts[0].set_position((0.5, 1.45))
    texts[1].set_position((-0.15, -1.6))
    texts[1].set_fontsize(5.5)
    texts[2].set_position((0.0, -1.6))
    texts[2].set_fontsize(5.5)
    texts[3].set_position((-0.8, -0.05))
    texts[3].set_fontsize(5)

    # D panel
    ax = fig.add_subplot(spec[2:6, 1])
    ax.text(-0.2, 0.94, 'D', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    features = ['EpitopeA', 'EpitopeB', 'EpitopeC', 'EpitopeD', 'EpitopeE',
                'Property1', 'Property2', 'Property3', 'Property4', 'Property5',
                'Nglycosylation', 'Receptor binding']

    features_newname = {'EpitopeA': 'BY_A', 'EpitopeB': 'BY_E', 'EpitopeC': 'BY_B', 'EpitopeD': 'BY_D',
                        'EpitopeE': 'BY_C',
                        'Property1': 'Hydrophobicity', 'Property2': 'Volume', 'Property3': 'Charge',
                        'Property4': 'Polarity',
                        'Property5': 'ASA', 'Nglycosylation': 'N-Gly', 'Receptor binding': 'RBS'}

    contributions = np.array([0.00324073, 0.08380397, 0.33116436, 0.25707573, 0.4314132, 0.22108154,
                              0.30920547, 0.44536322, 0.23724973, 0.47185928, 0.13346438, 0.25350255])

    contribution_per = [round((contribution / np.sum(contributions)) * 100, 2) for contribution in contributions]
    fea_con = dict(zip(contribution_per, features))
    con_sort = sorted(contribution_per, reverse=False)
    fea_sort = [features_newname[fea_con[con]] for con in con_sort]
    fea_num = list(np.arange(1, 13, 1))
    feature_size = [i * 5 for i in con_sort]

    colors = ['#A2D2BF', '#A2D2BF', '#A2D2BF', '#A2D2BF', '#A2D2BF',
              '#F69877', '#F69877', '#F69877', '#F69877', '#F69877',
              '#839DD1',
              '#FBCF4B']
    cor_con = dict(zip(contribution_per, colors))
    cor_sort = [cor_con[con] for con in con_sort]

    for i in range(len(fea_sort)):
        if i != 3:
            ax.text(x=con_sort[i] + 2.5 / 3, y=fea_num[i] - 0.15, s=str(con_sort[i]) + '%', fontsize=6)

    BY_lastnum = 3
    ax.text(x=con_sort[BY_lastnum] - 4, y=fea_num[BY_lastnum] - 0.15, s=str(con_sort[BY_lastnum]) + '%', fontsize=6)

    ax.scatter(con_sort, fea_num, s=feature_size, c=cor_sort, linewidths=0.5, edgecolors='grey', alpha=0.8, zorder=100)
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xticklabels(['0', '5%', '10%', '15%', '20%'])
    ax.set_xlabel('Contribution')
    ax.set_ylim(0.5, 12.5)
    ax.set_yticks(np.arange(1, 13, 1))
    ax.set_yticklabels(fea_sort)
    ax.tick_params(axis='x', labelsize=6, pad=1)
    ax.tick_params(axis='y', labelsize=6, pad=1)
    # ax.set_ylabel('Features')
    # ax.set_title('B/Yamagata', fontsize=10)
    ax.grid(ls=':', zorder=0, alpha=0.5)
    ax.set_axisbelow(True)

    size = 0.55
    con_group = [np.sum(contribution_per[:5]), np.sum(contribution_per[5:10]), np.sum(contribution_per[10]),
                 np.sum(contribution_per[11])]
    # color = ['#82B0D2', '#FFBE7A', '#8ECFC9', '#FA7F6F']
    color = ['#A2D2BF', '#F69877', '#839DD1', '#FBCF4B']
    BY_labels = ['Epitopes', 'Physicochemical \n Properties', 'Glycosylation \n sites', 'Receptor \n Binding Sites']
    pie_BY = fig.add_axes([0.508, 0.12, 0.16, 0.16])
    wedges, texts, autotexts = pie_BY.pie(con_group, radius=1.3, colors=color, autopct='%1.2f%%', shadow=False,
                                          pctdistance=0.81,
                                          labels=BY_labels, wedgeprops=dict(width=size, edgecolor='w'),
                                          textprops={"weight": "bold", "multialignment": "center"})

    for i, (text, autotext) in enumerate(zip(texts, autotexts)):
        size = con_group[i]
        proportion = size / sum(con_group)
        fontsize = 8 + 5 * proportion
        # text.set_fontsize(fontsize)
        autotext.set_fontsize(fontsize)
        autotext.set_color("w")
        autotext.set_text(f'{proportion:.1%}') 

    autotexts[0].set_position((0.15, 0.92))
    autotexts[0].set_rotation(0)
    # autotexts[1].set_text("47.1%")
    autotexts[1].set_rotation(-50)
    autotexts[2].set_fontsize(6)
    autotexts[2].set_rotation(-30)
    autotexts[3].set_fontsize(6)
    autotexts[3].set_rotation(0)

    texts[0].set_position((-0.35, 1.45))
    texts[1].set_position((-0.3, -1.6))
    texts[1].set_fontsize(5.5)
    texts[2].set_position((-0.25, -1.6))
    texts[2].set_fontsize(5.5)
    texts[3].set_position((-0.8, -0.05))
    texts[3].set_fontsize(5)


    # E panel
    def setAbsoluteTime(tree, dates):
        for i in tree.Objects:
            if i.is_leaf():
                i.absoluteTime = float(dates.loc[i.name, 'numeric date'])
            else:
                i.absoluteTime = float(dates.loc[i.traits['label'], 'numeric date'])
        tree.mostRecent = float(dates['numeric date'].max())
        return tree


    dates = pd.read_csv('./result/dates.tsv', sep='\t', index_col=0)

    tree_file = './result/timetree.nexus'

    df_color = pd.read_csv(
        './result/BV_node_color_paper.csv', index_col=0)

    tree = dendropy.Tree.get(path=tree_file, schema='nexus', extract_comment_metadata=True)
    treeString = tree.as_string(schema='newick', suppress_rooting=True, suppress_annotations=True,
                                annotations_as_nhx=False)

    ll = bt.loadNewick(StringIO(treeString), tip_regex='_([0-9\-]+)$', absoluteTime=False)
    ll = setAbsoluteTime(ll, dates)

    node_color = df_color.set_index('tree_name')['color'].to_dict()
    # print(node_color)
    treeHeight = ll.treeHeight

    ax = fig.add_subplot(spec[0:3, 2])
    ax.text(0, 0.97, 'E', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')

    y_attr = lambda k: -k.y
    x_attr = lambda k: k.absoluteTime


    def fetch_leaf_color(k):

        return node_color.get(k.name, 'black')


    color_func = lambda k: fetch_leaf_color(k) if k.branchType == 'leaf' else 'black'
    print(color_func)
    ll.plotTree(ax, width=0.3, x_attr=x_attr, y_attr=y_attr, alpha=1)
    ll.plotPoints(ax, x_attr=x_attr, y_attr=y_attr, size=2, colour=color_func, zorder=100, alpha=0.9)

    ID_name = {'EPI_ISL_2460_cluster7': 'B/Beijing/1/1987',
               'EPI_ISL_2821_cluster4': 'B/Shandong/7/1997', 'EPI_ISL_2342_cluster4': 'B/HongKong/330/2001',
               'EPI_ISL_30253_cluster4': 'B/Malaysia/2506/2004', 'EPI_ISL_28587_cluster1': 'B/Brisbane/60/2008',
               'EPI_ISL_257735_cluster5': 'B/Colorado/06/2017', 'EPI_ISL_7473160_cluster3': 'B/Washington/02/2019',
               'EPI_ISL_16967998_cluster2': 'B/Austria/1359417/2021'}
    vac_list = ['EPI_ISL_2460_cluster7',
                'EPI_ISL_2821_cluster4', 'EPI_ISL_2342_cluster4',
                'EPI_ISL_30253_cluster4', 'EPI_ISL_28587_cluster1',
                'EPI_ISL_257735_cluster5', 'EPI_ISL_7473160_cluster3',
                'EPI_ISL_16967998_cluster2']

    ID_color = {'EPI_ISL_2460_cluster7': 'r',
                'EPI_ISL_2821_cluster4': 'orange', 'EPI_ISL_2342_cluster4': 'green',
                'EPI_ISL_30253_cluster4': 'darkturquoise', 'EPI_ISL_28587_cluster1': 'darkviolet',
                'EPI_ISL_257735_cluster5': 'saddlebrown', 'EPI_ISL_7473160_cluster3': 'dodgerblue',
                'EPI_ISL_16967998_cluster2': 'magenta'}

    for leaf in ll.getExternal():
        if leaf.name in vac_list:
            print(leaf.name)
            cross_position = (leaf.absoluteTime, -leaf.y)
            ax.plot(*cross_position, 'D', markersize=3, zorder=100, c='saddlebrown', alpha=1)

    cluster_dict = {'BJ87': '#bf3a2b', 'SD97': '#3e58cf', 'BR08': '#4b8ec1', 'CO17': '#65ae96', 'WA19': '#8cbb69',
                    'COV20': '#dcab3c', 'AU21': '#e67932'}
    handles = []

    for k, v in cluster_dict.items():
        handles.append(
            Line2D([0], [0], lw=0, marker='o', markeredgecolor='none', markersize=3, markerfacecolor=v, label=k,
                   alpha=.9))
    for k, v in ID_name.items():
        handles.append(
            Line2D([0], [0], lw=0, marker='D', markeredgecolor='none', markersize=3, markerfacecolor=ID_color[k],
                   label=v, alpha=.9))
    ax.legend(handles=handles, frameon=False, fontsize=6, loc='upper left', bbox_to_anchor=(-.05, 1), labelspacing=0.2)

    ax.set_xlim(1983, 2030)
    ax.set_xticks(list(np.arange(1985, 2025, 5)) + [2024])
    ax.tick_params(axis='x', rotation=90, pad=1, )

    ax.set_ylim(-40 - ll.ySpan, 8)  ## set y limits
    ax.set_yticks([])
    ax.set_yticklabels([])
    # ax.set_title("B/Victoria")

    [ax.spines[loc].set_visible(False) for loc in ax.spines if loc != 'bottom']
    ax.grid(False)

    # F panel
    dates = pd.read_csv('./result/dates.tsv', sep='\t', index_col=0)

    tree_file = './result/timetree.nexus'

    df_color = pd.read_csv(
        './result/BY_node_color_paper.csv', index_col=0)

    tree = dendropy.Tree.get(path=tree_file, schema='nexus', extract_comment_metadata=True)
    treeString = tree.as_string(schema='newick', suppress_rooting=True, suppress_annotations=True,
                                annotations_as_nhx=False)

    ll = bt.loadNewick(StringIO(treeString), tip_regex='_([0-9\-]+)$', absoluteTime=False)
    ll = setAbsoluteTime(ll, dates)

    node_color = df_color.set_index('tree_name')['color'].to_dict()
    # print(node_color)
    treeHeight = ll.treeHeight

    # F panel
    ax = fig.add_subplot(spec[3:6, 2])
    ax.text(0.0, 0.90, 'F', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')

    y_attr = lambda k: -k.y
    x_attr = lambda k: k.absoluteTime


    def fetch_leaf_color(k):

        return node_color.get(k.name, 'black')


    color_func = lambda k: fetch_leaf_color(k) if k.branchType == 'leaf' else 'black'
    print(color_func)
    ll.plotTree(ax, width=0.3, x_attr=x_attr, y_attr=y_attr, alpha=1)
    ll.plotPoints(ax, x_attr=x_attr, y_attr=y_attr, size=2, colour=color_func, zorder=100, alpha=0.9)

    ID_name = {'EPI_ISL_20958_cluster3': 'B/Yamagata/16/1988',
               'EPI_ISL_20945_cluster3': 'B/Panama/45/1990', 'EPI_ISL_969_cluster2': 'B/Beijing/184/1993',
               'EPI_ISL_21113_cluster2': 'B/Sichuan/379/1999', 'EPI_ISL_6949688_cluster2': 'B/Shanghai/361/2002',
               'EPI_ISL_22470_cluster2': 'B/Belem/621/2005', 'EPI_ISL_22808_cluster2': 'B/Florida/4/2006',
               'EPI_ISL_76921_cluster1': 'B/Wisconsin/01/2010', 'EPI_ISL_138284_cluster2': 'B/Massachusetts/02/2012',
               'EPI_ISL_165881_cluster1': 'B/Phuket/3073/2013'}
    vac_list = ['EPI_ISL_20958_cluster3',
                'EPI_ISL_20945_cluster3', 'EPI_ISL_969_cluster2',
                'EPI_ISL_21113_cluster2', 'EPI_ISL_6949688_cluster2',
                'EPI_ISL_22470_cluster2', 'EPI_ISL_22808_cluster2',
                'EPI_ISL_76921_cluster1', 'EPI_ISL_138284_cluster2',
                'EPI_ISL_165881_cluster1']

    ID_color = {'EPI_ISL_20958_cluster3': 'k',
                'EPI_ISL_20945_cluster3': 'r', 'EPI_ISL_969_cluster2': 'dodgerblue',
                'EPI_ISL_21113_cluster2': 'b', 'EPI_ISL_6949688_cluster2': 'darkturquoise',
                'EPI_ISL_22470_cluster2': 'darkviolet', 'EPI_ISL_22808_cluster2': 'saddlebrown',
                'EPI_ISL_76921_cluster1': 'magenta', 'EPI_ISL_138284_cluster2': 'orange',
                'EPI_ISL_165881_cluster1': 'green'}

    for leaf in ll.getExternal():
        if leaf.name in vac_list:
            print(leaf.name)
            cross_position = (leaf.absoluteTime, -leaf.y)
            ax.plot(*cross_position, 'D', markersize=3, zorder=100, c='saddlebrown')

    cluster_dict = {'YA88': '#4988c5', 'BJ93': '#7eb876', 'WI10': '#cbb742'}
    handles = []
    for k, v in cluster_dict.items():
        handles.append(
            Line2D([0], [0], lw=0, marker='o', markeredgecolor='none', markersize=3, markerfacecolor=v, label=k,
                   alpha=.9))
    for k, v in ID_name.items():
        handles.append(
            Line2D([0], [0], lw=0, marker='D', markeredgecolor='none', markersize=3, markerfacecolor=ID_color[k],
                   label=v, alpha=.9))
    ax.legend(handles=handles, frameon=False, fontsize=6, loc='upper left', bbox_to_anchor=(-.05, 1), labelspacing=0.2)

    ax.set_xlim(1983, 2026)
    ax.set_xticks(list(np.arange(1985, 2021, 5)))
    ax.tick_params(axis='x', rotation=90, pad=1, )

    ax.set_ylim(-40 - ll.ySpan, 8)  ## set y limits
    ax.set_yticks([])
    ax.set_yticklabels([])
    # ax.set_title("B/Yamagata")

    [ax.spines[loc].set_visible(False) for loc in ax.spines if loc != 'bottom']
    ax.grid(False)
    fig.get_layout_engine().set(w_pad=10 / 300, h_pad=1.5 / 300, wspace=0.01, hspace=0.02)

plt.savefig(
    r"./figure/fig3.pdf")
plt.show()
