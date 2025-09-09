# -*- coding: utf-8 -*-
# @Time    : 2025/4/1 下午2:26
# @Author  : Hanwenjie
# @project : BY_dn_ds_cluster_MEME.py
# @File    : fig_s2.py
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
    r"./result/BV_model_performance.csv", index_col=0)
df_BY = pd.read_csv(
    r"./result/BY_model_performance.csv", index_col=0)

# model comparison
# specify unit conversion to cm
cm = 2.54

# plotting
fig = plt.figure(figsize=(14/cm, 7/cm), dpi=300, layout="compressed")
spec = fig.add_gridspec(1,2)
trans = ScaledTranslation(-30/300, 30/300, fig.dpi_scale_trans)
with mpl.rc_context({'font.family': 'Arial', 'font.size': 7, 'hatch.linewidth': .5, 'lines.linewidth': .3, 'patch.linewidth': .3, 'xtick.labelsize': 6, 'ytick.labelsize': 6}):

    # Panel A
    ax = fig.add_subplot(spec[0, 0])
    ax.text(-0.1, 0.97, 'A', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    # BV cross-validation evaluation
    models = ("XGBoost", "LightGBM", "RF", "SVM")
    metrics = {
        'AUC': df_BV['auc_cv'],
        'Accuracy': df_BV['accuracy_cv'],
        'Precision': df_BV['precision_cv'],
        'Recall': df_BV['recall_cv'],
        'F1 Score': df_BV['f1-score_cv'],
    }

    # colors = ['#82B0D2', '#FA7F6F', '#FFBE7A', '#8ECFC9', '#BEB8DC']
    # colors = ['#4180B2', '#FBCF4B', '#F69877', '#87D4D6', '#7B8391']
    colors = ['#839DD1', '#FBCF4B', '#F69877', '#A2D2BF', '#7B8391']
    x = np.arange(len(models))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0

    i = 0
    for attribute, measurement in metrics.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[i], alpha=1, zorder=5)
        multiplier += 1
        i += 1
        # break

    ax.tick_params(which='major', direction='out', bottom=False)
    ax.set_xticks(x + 2 * width, models)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0,1.1,0.2))
    ax.tick_params(axis='x',pad=0)
    ax.tick_params(axis='y', pad=1)
    # ax.set_title('B/Victoria',pad=10,)
    ax.grid(axis='y', ls='--', zorder=0, alpha=0.5)
    ax.spines['bottom'].set_zorder(10)
    ax.legend(['AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score'], frameon=False, ncols=5, fontsize=6,
              loc='lower left', bbox_to_anchor=(-0.02, 1), columnspacing=.9, handletextpad=0.25,handlelength=0.7)
    ax.set_xlabel('Model')
    ax.set_ylabel('Performance')
    # ax.set_axisbelow(True)

    # Panel B
    ax = fig.add_subplot(spec[0, 1])
    ax.text(-0.1, 0.97, 'B', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    # BY evaluation
    # cross-validation
    models = ("XGBoost", "LightGBM", "RF", "SVM")
    metrics = {
        'AUC': df_BY['auc_cv'],
        'Accuracy': df_BY['accuracy_cv'],
        'Precision': df_BY['precision_cv'],
        'Recall': df_BY['recall_cv'],
        'F1 Score': df_BY['f1-score_cv'],
    }

    colors = ['#839DD1', '#FBCF4B', '#F69877', '#A2D2BF', '#7B8391']
    # colors = ['#4180B2', '#FBCF4B', '#F69877', '#87D4D6', '#7B8391']
    x = np.arange(len(models))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0

    i = 0
    for attribute, measurement in metrics.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[i], alpha=1, zorder=5)
        multiplier += 1
        i += 1
        # break

    ax.tick_params(which='major', direction='out', bottom=False)
    ax.set_xticks(x + 2 * width, models)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(axis='x',  pad=0)
    ax.tick_params(axis='y', pad=1)
    # ax.set_title('B/Yamagata',pad=10)
    ax.grid(axis='y', ls='--', zorder=0, alpha=0.5)
    # set zorder parameter for x-axis
    ax.spines['bottom'].set_zorder(10)
    ax.set_xlabel('Model')
    ax.set_ylabel('Performance')
    # ax.set_axisbelow(False)
    ax.legend(['AUC', 'Accuracy','Precision','Recall','F1 Score'], frameon=False, ncols=5, fontsize=6,
              loc='lower left', bbox_to_anchor=(-0.02, 1), columnspacing=.9, handletextpad=0.25,handlelength=0.7)
plt.savefig(r"./figure/fig_s2.pdf")
plt.show()
