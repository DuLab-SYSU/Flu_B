# -*- coding: utf-8 -*-
# @Time    : 2025/7/8 上午9:38
# @Author  : Hanwenjie
# @project : BVcross_Ngly.py
# @File    : fig_s7.py
# @IDE     : PyCharm
# @REMARKS : description text
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

# specify unit conversion to cm
cm = 2.54

fig = plt.figure(figsize=(18/cm, 7.7/cm), dpi=300, layout="constrained")
spec = fig.add_gridspec(2,4)
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
    #按cluster时间顺序排序
    df_log_reordered = df_log.reindex(index=new_row_cluster,columns=new_col_cluster)

    #similarity ratio predict dataset
    df_ratio = df.pivot(index="cluster1", columns="cluster2", values="similarity_ratio")
    # 按cluster时间顺序排序
    df_ratio_reordered = df_ratio.reindex(index=new_row_cluster, columns=new_col_cluster)

    # similarity ratio in true HI dataset
    df_ratio_HI = df_HI.pivot(index="cluster1", columns="cluster2", values="similarity_ratio")
    # 按cluster时间顺序排序
    df_ratio_HI_reordered = df_ratio_HI.reindex(index=new_row_cluster, columns=new_col_cluster)

    # import antigenic distance between clusters
    df_distance = pd.read_csv(r"./result/"
                              r"BV_cluster_distance.csv", index_col=0)
    df_distance_cross = df_distance.pivot(index="cluster1", columns="cluster2", values="anti_distance")
    df_distance_reordered = df_distance_cross.reindex(index=new_row_cluster,columns=new_col_cluster)

    x_text = -0.31
    y_text = 0.85
    # A panel
    ax = fig.add_subplot(spec[0, 0])
    ax.text(x_text, y_text, 'a', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',fontfamily='Arial')

    heatmap = sns.heatmap(data=df_log_reordered, square=True,
                          vmin=-6, vmax=6,
                          cmap="RdBu_r", center=0,
                          annot=True, fmt=".2f",annot_kws={'fontsize':4},
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
    ax.text(x_text, y_text, 'b', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom', fontfamily='Arial')
    heatmap = sns.heatmap(data=df_ratio_reordered, square=True,
                          vmin=0, vmax=1,
                          cmap="OrRd", center=0.5,
                          annot=True, fmt=".2f",annot_kws={'fontsize':4},
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
    ax.text(x_text, y_text, 'c', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom', fontfamily='Arial')
    mask = df_ratio_HI_reordered.isna()
    heatmap = sns.heatmap(data=df_ratio_HI_reordered, square=True,
                          vmin=0, vmax=1,
                          cmap="OrRd", center=0.5,
                          annot=True, fmt=".2f",annot_kws={'fontsize':4},
                          cbar_kws={'shrink': 0.5, 'aspect': 20, 'ticks': [0, 0.5, 1]},
                          ax=ax, )
    for (i, j), val in np.ndenumerate(df_ratio_HI_reordered.values):
        if pd.isna(val):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, hatch='///', edgecolor='lightgray', linewidth=0.0))
    # ax.set_title('B/Victoria-Similarity rate', fontsize=7)
    ax.set_xticklabels(['BJ87', 'SD97', 'BR08', 'CO17', 'WA19', 'COV20', 'AU21'])
    ax.set_yticklabels(['BJ87', 'SD97', 'BR08', 'CO17', 'WA19', 'COV20', 'AU21'])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=90)

    # D
    ax = fig.add_subplot(spec[0, 3])
    ax.text(x_text, y_text, 'd', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',fontfamily='Arial')

    mask = df_distance_reordered.isna()

    heatmap = sns.heatmap(data=df_distance_reordered, mask=mask, square=True,
                          vmin=0, vmax=6,
                          cmap="OrRd", center=3,
                          annot=True, fmt=".2f",annot_kws={'fontsize':4},
                          cbar_kws={'shrink': 0.5, 'aspect': 20, 'ticks': [0, 3, 6]},
                          ax=ax)

    for (i, j), val in np.ndenumerate(df_distance_reordered.values):
        if pd.isna(val):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, hatch='///', edgecolor='lightgray', linewidth=0.0))
            # ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='lightgray', edgecolor='none'))
    # ax.set_title('B/Victoria')
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
    # 按cluster时间顺序排序
    df_log_reordered = df_log.reindex(index=new_row_cluster, columns=new_col_cluster)

    # similarity ratio predict dataset
    df_ratio = df.pivot(index="cluster1", columns="cluster2", values="similarity_ratio")
    # 按cluster时间顺序排序
    df_ratio_reordered = df_ratio.reindex(index=new_row_cluster, columns=new_col_cluster)

    # similarity ratio in true HI dataset
    df_ratio_HI = df_HI.pivot(index="cluster1", columns="cluster2", values="similarity_ratio")
    # 按cluster时间顺序排序
    df_ratio_HI_reordered = df_ratio_HI.reindex(index=new_row_cluster, columns=new_col_cluster)

    # import antigenic distance between clusters
    df_distance = pd.read_csv(r"./result/"
                              r"BY_cluster_distance.csv", index_col=0)
    df_distance_cross = df_distance.pivot(index="cluster1", columns="cluster2", values="anti_distance")
    df_distance_reordered = df_distance_cross.reindex(index=new_row_cluster, columns=new_col_cluster)

    # e panel
    ax = fig.add_subplot(spec[1, 0])
    ax.text(x_text, y_text, 'e', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom', fontfamily='Arial')

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

    # F panel
    ax = fig.add_subplot(spec[1, 1])
    ax.text(x_text, y_text, 'f', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom', fontfamily='Arial')
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

    # G panel
    ax = fig.add_subplot(spec[1, 2])
    ax.text(x_text, y_text, 'g', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom', fontfamily='Arial')
    mask = df_ratio_HI_reordered.isna()
    heatmap = sns.heatmap(data=df_ratio_HI_reordered, square=True,
                          vmin=0, vmax=1,
                          cmap="OrRd", center=0.5,
                          annot=True, fmt=".2f", annot_kws={'fontsize': 5},
                          cbar_kws={'shrink': 0.5, 'aspect': 20, 'ticks': [0, 0.5, 1]},
                          ax=ax, )
    for (i, j), val in np.ndenumerate(df_distance_reordered.values):
        if pd.isna(val):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, hatch='///', edgecolor='lightgray', linewidth=0.0))

    # ax.set_title('B/Yamagata')
    ax.set_xticklabels(['YA88', 'BJ93', 'WI10'])
    ax.set_yticklabels(['YA88', 'BJ93', 'WI10'])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='y', rotation=0)
    ax.tick_params(axis='x', rotation=90)

    # H panel
    ax = fig.add_subplot(spec[1, 3])
    ax.text(x_text, y_text, 'h', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    mask = df_distance_reordered.isna()
    heatmap = sns.heatmap(data=df_distance_reordered, square=True,
                          vmin=1, vmax=3,
                          cmap="OrRd", center=2,
                          annot=True, fmt=".2f", annot_kws={'fontsize': 5},
                          cbar_kws={'shrink': 0.5, 'aspect': 20, 'ticks': [1, 2, 3]},
                          ax=ax, )
    for (i, j), val in np.ndenumerate(df_distance_reordered.values):
        if pd.isna(val):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, hatch='///', edgecolor='lightgray', linewidth=0.0))

    # ax.set_title('B/Yamagata')
    ax.set_xticklabels(['YA88', 'BJ93', 'WI10'])
    ax.set_yticklabels(['YA88', 'BJ93', 'WI10'])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='y', rotation=0)
    ax.tick_params(axis='x', rotation=90)

    fig.get_layout_engine().set(w_pad=3/300, h_pad=3/300, hspace=0.01, wspace=0.01)
plt.savefig(r"./figure/fig_s7.pdf")
plt.show()