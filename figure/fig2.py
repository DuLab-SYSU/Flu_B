# -*- coding: utf-8 -*-
# @Time    : 2025/4/1 下午2:26
# @Author  : Hanwenjie
# @project : BY_dn_ds_cluster_MEME.py
# @File    : fig2_v10.py
# @IDE     : PyCharm
# @REMARKS : Description text
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
    r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BV_gasaid/BV_region_seq/data_process/final/"
    r"figure_data/model_performance/BV_model_performance.csv", index_col=0)
df_BY = pd.read_csv(
    r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BV_gasaid/BV_region_seq/data_process/final/"
    r"figure_data/model_performance/BY_model_performance.csv", index_col=0)
# set centimeter conversion
cm = 2.54

# plot figure
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
    # BV cross-validation evaluation
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
    ax.grid(axis='y', ls='--', zorder=0, alpha=0.5)
    ax.spines['bottom'].set_zorder(10)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Performance')

    # B panel
    ax = fig.add_subplot(spec[0:2, 1])
    ax.text(-0.2, 0.97, 'B', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    # BY cross-validation evaluation
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
    ax.grid(axis='y', ls='--', zorder=0, alpha=0.5)
    ax.spines['bottom'].set_zorder(10)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Performance')

    # C panel
    ax = fig.add_subplot(spec[2:6, 0])
    ax.text(-0.2, 0.94, 'C', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    # define features and new names
    features = ['EpitopeA', 'EpitopeB', 'EpitopeC', 'EpitopeD', 'EpitopeE',
                'Property1', 'Property2', 'Property3', 'Property4', 'Property5',
                'Nglycosylation', 'Receptor binding']
    features_newname = {'EpitopeA': 'BV_E', 'EpitopeB': 'BV_A', 'EpitopeC': 'BV_D', 'EpitopeD': 'BV_C',
                        'EpitopeE': 'BV_B',
                        'Property1': 'Hydrophobicity', 'Property2': 'Volume', 'Property3': 'Charge',
                        'Property4': 'Polarity',
                        'Property5': 'ASA', 'Nglycosylation': 'N-Gly', 'Receptor binding': 'RBS'}

    # contribution values
    contributions = np.array([0.313, 3.60387486, 0.55467476, 1.90445524, 0.65409724, 0.85292539,
                              0.59821007, 0.76302093, 1.74082968, 0.56377183, 0.64845957, 1.02104887])

    contribution_per = [round((contribution / np.sum(contributions)) * 100, 2) for contribution in contributions]
    fea_con = dict(zip(contribution_per, features))
    con_sort = sorted(contribution_per, reverse=False)
    fea_sort = [features_newname[fea_con[con]] for con in con_sort]
    fea_num = list(np.arange(1, 13, 1))
    feature_size = [i * 5 for i in con_sort]

    # define colors for each feature
    colors = ['#A2D2BF', '#A2D2BF', '#A2D2BF', '#A2D2BF', '#A2D2BF',
              '#F69877', '#F69877', '#F69877', '#F69877', '#F69877',
              '#839DD1',
              '#FBCF4B']
    cor_con = dict(zip(contribution_per, colors))
    cor_sort = [cor_con[con] for con in con_sort]

    # annotate each feature's contribution
    for i in range(len(fea_sort) - 1):
        ax.text(x=con_sort[i] + 1.1, y=fea_num[i] - 0.15, s=str(con_sort[i]) + '%', fontsize=6)

    # annotate last feature separately to avoid clipping
    BV_lastnum = len(fea_sort) - 1
    ax.text(x=con_sort[BV_lastnum] - 7.5, y=fea_num[BV_lastnum] - 0.15, s=str(con_sort[BV_lastnum]) + '%', fontsize=6)
    # set plotting details
    ax.scatter(con_sort, fea_num, s=feature_size, c=cor_sort, linewidths=0.5, edgecolors='grey', alpha=0.8, zorder=100)
    ax.set_xlim(0, 30)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xticklabels(['0', '5%', '10%', '15%', '20%', '25%', '30%'])
    ax.set_xlabel('Contribution')
    ax.set_ylim(0.5, 12.5)
   
