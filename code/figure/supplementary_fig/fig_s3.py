# -*- coding: utf-8 -*-
# @Time    : 2025/4/1 下午2:26
# @Author  : Hanwenjie
# @project : BY_dn_ds_cluster_MEME.py
# @File    : fig_s3.py
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

df_BV_best = df_BV[df_BV['model']=='lightgbm'].copy()
df_BY_best = df_BY[df_BY['model']=='xgboost'].copy()
df_test = pd.concat([df_BV_best,df_BY_best],ignore_index=True)
df_test.reset_index(inplace=True,drop=True)

# specify unit conversion to cm
cm = 2.54

# plot
fig = plt.figure(figsize=(7/cm, 7/cm), dpi=300, layout="constrained")
spec = fig.add_gridspec(1,1)
trans = ScaledTranslation(-30/300, 30/300, fig.dpi_scale_trans)
with mpl.rc_context({'font.family': 'Arial', 'font.size': 7, 'hatch.linewidth': .5, 'lines.linewidth': .3, 'patch.linewidth': .3, 'xtick.labelsize': 6, 'ytick.labelsize': 6}):
    # A panel
    ax = fig.add_subplot(spec[0:1, 0])
    ax.text(0.0, 1.0, '', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    # BV cross-validation evalutation
    lineage = ("PREDAC-BV", "PREDAC-BY")
    metrics = {
        'AUC': df_test['auc_test'],
        'Accuracy': df_test['accuracy_test'],
        'Precision': df_test['precision_test'],
        'Recall': df_test['recall_test'],
        'F1 Score': df_test['f1-score_test'],
    }

    # colors = ['#82B0D2', '#FA7F6F', '#FFBE7A', '#8ECFC9', '#BEB8DC']
    colors = ['#839DD1', '#FBCF4B', '#F69877', '#A2D2BF', '#7B8391']
    # x = np.arange(len(lineage))  # the label locations
    x = np.array([0.3,1.3])  # the label locations
    width = 0.1  # the width of the bars
    multiplier = 0

    i = 0
    for attribute, measurement in metrics.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[i], zorder=5,alpha=1)
        multiplier += 1
        i += 1
        # break

    ax.tick_params(which='major', direction='out', bottom=False)
    ax.set_xlim(0, 2)
    ax.set_xticks(x+2*width, lineage)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.tick_params(axis='x', pad=0)
    ax.tick_params(axis='y', pad=1)
    ax.set_xlabel("Model")
    ax.set_ylabel("Performance")
    # ax.set_title('Model performance of independent test', pad=10, )
    ax.grid(axis='y', ls='--', zorder=0, alpha=0.5)
    ax.spines['bottom'].set_zorder(10)
    ax.legend(['AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score'], frameon=False, ncols=5, fontsize=6,
              loc='lower left', bbox_to_anchor=(-0.03, 0.98), columnspacing=.3, handletextpad=0.25, handlelength=1.5)
    # ax.set_axisbelow(True)
plt.savefig(r"./figure/fig_s3.pdf")
plt.show()