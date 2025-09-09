# -*- coding: utf-8 -*-
# @Time    : 2025/3/22 下午6:12
# @Author  : Hanwenjie
# @project : BY_dn_ds_cluster_MEME.py
# @File    : fig_s19.py
# @IDE     : PyCharm
# @REMARKS : Description text
import pandas as pd
import baltic as bt
import dendropy
from io import StringIO
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from matplotlib.transforms import ScaledTranslation
mpl.rcParams['pdf.fonttype']=42

specify unit conversion to cm
cm = 2.54

fig = plt.figure(figsize=(18/cm, 11/cm), dpi=300, layout="constrained")
spec = fig.add_gridspec(2,3)
trans = ScaledTranslation(-30/300, 30/300, fig.dpi_scale_trans)
with mpl.rc_context({'font.family': 'Arial', 'font.size': 7, 'hatch.linewidth': .5, 'lines.linewidth': .3, 'patch.linewidth': .3, 'xtick.labelsize': 6, 'ytick.labelsize': 6}):

    df = pd.read_csv(
        './result/MCL_eval.csv', index_col=0)
    df_HI = pd.read_csv(
        './result/MCL_ratio_trueHI.csv', index_col=0)
    # log ratio predict dataset
    df_log = df.pivot(index="cluster1", columns="cluster2", values="mean_ratio")

    new_row_cluster = ['cluster7','cluster4','cluster1','cluster5','cluster3','cluster6','cluster2']
    new_col_cluster = ['cluster7', 'cluster4', 'cluster1', 'cluster5', 'cluster3', 'cluster6', 'cluster2']
    # sort by cluster chronological order
    df_log_reordered = df_log.reindex(index=new_row_cluster,columns=new_col_cluster)

    #similarity ratio predict dataset
    df_ratio = df.pivot(index="cluster1", columns="cluster2", values="similarity_ratio")
    # sort by cluster chronological order
    df_ratio_reordered = df_ratio.reindex(index=new_row_cluster, columns=new_col_cluster)

    # similarity ratio in true HI dataset
    df_ratio_HI = df_HI.pivot(index="cluster1", columns="cluster2", values="similarity_ratio")
    # sort by cluster chronological order
    df_ratio_HI_reordered = df_ratio_HI.reindex(index=new_row_cluster, columns=new_col_cluster)

    # A panel
    ax = fig.add_subplot(spec[0, 0])
    ax.text(-0.18, 0.93, 'A', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',fontfamily='Arial')

    heatmap = sns.heatmap(data=df_log_reordered, square=True,
                          vmin=-6, vmax=6,
                          cmap="RdBu_r", center=0,
                          annot=True, fmt=".2f",annot_kws={'fontsize':5},
                          cbar_kws={'shrink': 0.5, 'aspect': 20, 'ticks': [-6, 0, 6]},
                          ax=ax, )
    # ax.set_title('B/Victoria-log ratio', fontsize=7)
    ax.set_xticklabels(['BJ87', 'SD97', 'BR08', 'CO17', 'WA19', 'COV20', 'AU21'])
    ax.set_yticklabels(['BJ87', 'SD97', 'BR08', 'CO17', 'WA19', 'COV20', 'AU21'])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=90)

    # B panel
    ax = fig.add_subplot(spec[0, 1])
    ax.text(-0.18, 0.93, 'B', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom', fontfamily='Arial')
    heatmap = sns.heatmap(data=df_ratio_reordered, square=True,
                          vmin=0, vmax=1,
                          cmap="OrRd", center=0.5,
                          annot=True, fmt=".2f",annot_kws={'fontsize':5},
                          cbar_kws={'shrink': 0.5, 'aspect': 20, 'ticks': [0, 0.5, 1]},
                          ax=ax, )
    # ax.set_title('B/Victoria')
    ax.set_xticklabels(['BJ87', 'SD97', 'BR08', 'CO17', 'WA19', 'COV20', 'AU21'])
    ax.set_yticklabels(['BJ87', 'SD97', 'BR08', 'CO17', 'WA19', 'COV20', 'AU21'])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=90)

    # C panel
    ax = fig.add_subplot(spec[0, 2])
    ax.text(-0.18, 0.93, 'C', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom', fontfamily='Arial')
    heatmap = sns.heatmap(data=df_ratio_HI_reordered, square=True,
                          vmin=0, vmax=1,
                          cmap="OrRd", center=0.5,
                          annot=True, fmt=".2f",annot_kws={'fontsize':5},
                          cbar_kws={'shrink': 0.5, 'aspect': 20, 'ticks': [0, 0.5, 1]},
                          ax=ax, )
    # ax.set_title('B/Victoria-Similarity rate', fontsize=7)
    ax.set_xticklabels(['BJ87', 'SD97', 'BR08', 'CO17', 'WA19', 'COV20', 'AU21'])
    ax.set_yticklabels(['BJ87', 'SD97', 'BR08', 'CO17', 'WA19', 'COV20', 'AU21'])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=90)


    #B/Yamagata heatmap
    df = pd.read_csv(
        './result/MCL_eval.csv', index_col=0)
    df_HI = pd.read_csv(
        './result/MCL_ratio_trueHI.csv', index_col=0)

    # log ratio predict dataset
    df_log = df.pivot(index="cluster1", columns="cluster2", values="mean_ratio")

    new_row_cluster = ['cluster3', 'cluster2', 'cluster1']
    new_col_cluster = ['cluster3', 'cluster2', 'cluster1']
    # sort by cluster chronological order
    df_log_reordered = df_log.reindex(index=new_row_cluster, columns=new_col_cluster)

    # similarity ratio predict dataset
    df_ratio = df.pivot(index="cluster1", columns="cluster2", values="similarity_ratio")
    # sort by cluster chronological order
    df_ratio_reordered = df_ratio.reindex(index=new_row_cluster, columns=new_col_cluster)

    # similarity ratio in true HI dataset
    df_ratio_HI = df_HI.pivot(index="cluster1", columns="cluster2", values="similarity_ratio")
    # sort by cluster chronological order
    df_ratio_HI_reordered = df_ratio_HI.reindex(index=new_row_cluster, columns=new_col_cluster)

    # D panel
    ax = fig.add_subplot(spec[1, 0])
    ax.text(-0.18, 0.93, 'D', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom', fontfamily='Arial')

    heatmap = sns.heatmap(data=df_log_reordered, square=True,
                          vmin=-2, vmax=2,
                          cmap="RdBu_r", center=0,
                          annot=True, fmt=".2f", annot_kws={'fontsize': 5},
                          cbar_kws={'shrink': 0.5, 'aspect': 20, 'ticks': [-2, 0, 2]},
                          ax=ax, )
    # ax.set_title('B/Yamagata')
    ax.set_xticklabels(['YA88', 'BJ93', 'WI10'])
    ax.set_yticklabels(['YA88', 'BJ93', 'WI10'])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='y', rotation=0)
    ax.tick_params(axis='x', rotation=90)

    # E panel
    ax = fig.add_subplot(spec[1, 1])
    ax.text(-0.18, 0.93, 'E', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom', fontfamily='Arial')
    heatmap = sns.heatmap(data=df_ratio_reordered, square=True,
                          vmin=0, vmax=1,
                          cmap="OrRd", center=0.5,
                          annot=True, fmt=".2f", annot_kws={'fontsize': 5},
                          cbar_kws={'shrink': 0.5, 'aspect': 20, 'ticks': [0, 0.5, 1]},
                          ax=ax, )
    # ax.set_title('B/Yamagata')
    ax.set_xticklabels(['YA88', 'BJ93', 'WI10'])
    ax.set_yticklabels(['YA88', 'BJ93', 'WI10'])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='y', rotation=0)
    ax.tick_params(axis='x', rotation=90)

    # F panel
    ax = fig.add_subplot(spec[1, 2])
    ax.text(-0.18, 0.93, 'F', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom', fontfamily='Arial')
    heatmap = sns.heatmap(data=df_ratio_HI_reordered, square=True,
                          vmin=0, vmax=1,
                          cmap="OrRd", center=0.5,
                          annot=True, fmt=".2f", annot_kws={'fontsize': 5},
                          cbar_kws={'shrink': 0.5, 'aspect': 20, 'ticks': [0, 0.5, 1]},
                          ax=ax, )
    # ax.set_title('B/Yamagata')
    ax.set_xticklabels(['YA88', 'BJ93', 'WI10'])
    ax.set_yticklabels(['YA88', 'BJ93', 'WI10'])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='y', rotation=0)
    ax.tick_params(axis='x', rotation=90)
plt.savefig(r"./figure/fig_s19.pdf")
plt.show()




