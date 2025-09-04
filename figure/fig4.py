# -*- coding: utf-8 -*-
# @Time    : 2025/3/22 下午6:01
# @Author  : Hanwenjie
# @project : BY_dn_ds_cluster_MEME.py
# @File    : fig4_v9.py
# @IDE     : PyCharm
# @REMARKS : 说明文字
import os
import math
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

# 指定厘米换算单位
cm = 2.54
# SR rate
df = pd.read_csv(r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BV_gasaid/BV_region_seq/data_process/unfill/"
                 r"figure_data/DNA/dn_ds/dn_ds_site/sample1/figure/dn_ds_SR_time.csv",index_col=0)

# df.dropna(subset=['BV_SR','BY_SR'],inplace=True)
df = df[df['time_plot'] > 2006].copy()
df.reset_index(inplace=True,drop=True)

# 总dn/ds
df_all_site = pd.read_csv(
    r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BV_gasaid/BV_region_seq/data_process/unfill/"
    r"figure_data/DNA/dn_ds/dn_ds_site/sample1/figure/dn_ds_total.csv", index_col=0)
# 总internal, external dn/ds
df_all_branch = pd.read_csv(
    r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BV_gasaid/BV_region_seq/data_process/unfill/"
    r"figure_data/DNA/dn_ds/dn_ds_site/sample1/figure/dn_ds_branch_total.csv", index_col=0)

# 数据集处理
df_time = pd.read_csv(
    r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BV_gasaid/BV_region_seq/data_process/unfill/"
    r"figure_data/DNA/dn_ds/dn_ds_site/sample1/figure/dn_ds_time.csv", index_col=0)

# U检验数据
df_u = pd.read_csv(
    r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BV_gasaid/BV_region_seq/data_process/unfill/"
    r"figure_data/DNA/dn_ds/dn_ds_site/sample1/figure/dn_ds_time_site_test.csv")

index_values = df_u.loc[df_u['time'] == '2006-U'].index[0]
fig_column = [str1 + '_' + str2 + '_' + 'test' for str1 in ['dn', 'ds'] for str2 in ['all', 'HA1', 'epi', 'noepi']]
fig_column.insert(0, 'time')
df_figure = df_u.loc[index_values:df_u.shape[0] - 2, fig_column].copy()
df_figure.reset_index(inplace=True, drop=True)
df_figure.fillna(0, inplace=True)
df_figure.replace('positive', 1, inplace=True)
df_figure.replace('negative', -1, inplace=True)
df_figure.set_index('time', drop=True, inplace=True)
df_figure = df_figure.T
# print(df_figure)

# 序列数量数据
df_seq_count = pd.read_csv(
    r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BV_gasaid/BV_region_seq/data_process/unfill/"
    r"figure_data/DNA/dn_ds/dn_ds_site/sample1/figure/seq_count.csv", index_col=0)
df_seq_count = df_seq_count[(df_seq_count['time_plot'] >= 2006) & (df_seq_count['time_plot'] <= 2020.5)].copy()
df_seq_count.reset_index(inplace=True, drop=True)

# cluster百分比图
df_BV_cluster = pd.read_csv(
    r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BV_gasaid/BV_region_seq/data_process/unfill/"
    r"figure_data/DNA/dn_ds/dn_ds_site/sample1/figure/BV_cluster_proportion.csv", index_col=0)
df_BV_cluster = df_BV_cluster[(df_BV_cluster['time_plot'] >= 2006) & (df_BV_cluster['time_plot'] <= 2020.5)].copy()
df_BV_cluster.reset_index(inplace=True, drop=True)

df_BY_cluster = pd.read_csv(
    r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BY_gasaid/BY_region_seq/data_process/unfill/"
    r"figure_data/DNA/dn_ds/dn_ds_site/sample1/figure/BY_cluster_proportion.csv", index_col=0)
df_BY_cluster = df_BY_cluster[(df_BY_cluster['time_plot'] >= 2006) & (df_BY_cluster['time_plot'] <= 2020.5)].copy()
df_BY_cluster.reset_index(inplace=True, drop=True)

# 画图
fig = plt.figure(figsize=(18.4 / cm, 8 / cm), dpi=300, layout="constrained")
spec = fig.add_gridspec(2, 2)
trans = ScaledTranslation(-30 / 300, 30 / 300, fig.dpi_scale_trans)
with mpl.rc_context(
        {'font.family': 'Arial', 'font.size': 7, 'hatch.linewidth': .5, 'lines.linewidth': .3, 'patch.linewidth': .3,
         'xtick.labelsize': 6, 'ytick.labelsize': 6}):
    # A panel
    ax = fig.add_subplot(spec[0, 0])
    ax.text(-0.08, 0.95, 'A', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    ax.plot(df['time_plot'], df['BV_SR_abs'], marker='o', markersize=2.3, color='#62B197',
            linewidth=1.5, zorder=100,
            label='B/Victoria', alpha=1)
    ax.plot(df['time_plot'], df['BY_SR_abs'], marker='o', markersize=2.3, color='#E18E6D',
            linewidth=1.5, zorder=100,
            label='B/Yamagata', alpha=1)
    ax.set_ylim(0, 10)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_ylabel('Substitution rate ($\\times 10^{-3}$)')
    ax.set_title('Substitution rates', fontsize=7)
    ax.set_xlim(2006, 2023)
    ax.set_xticks(list(range(2006, 2024)))
    ax.tick_params(axis='x', labelsize=6, rotation=90, pad=1)
    ax.tick_params(axis='y', labelsize=6, pad=1)
    # ax.set_xlabel('Year')
    # ax.axvline(x=2013, zorder=1, color='#bf3a2b', ls='--', linewidth=1.5)
    ax.grid(ls='-', zorder=0, alpha=0.5)
    ax.legend(ncol=2, frameon=False, loc='upper left', fontsize=6)
    ax.axvspan(2020 + 1 / 12, 2021 + 8 / 12, facecolor='#ef8c3b', edgecolor='none', alpha=.2)
    ax.set_axisbelow(True)

    # panel D,E,F,G
    index_ = 'CD'
    for i, cat in enumerate(['external', 'internal']):
        # 指定数据集
        df = df_time.copy()

        # ABC panel
        if cat != 'all':
            ax = fig.add_subplot(spec[1, i])
            ax.text(-0.08, 0.95, index_[i], transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
                    fontfamily='Arial')

            # 指定画图的列
            BV_rate = 'BV_{rate_type}_dn/ds'.format(rate_type=cat)
            BY_rate = 'BY_{rate_type}_dn/ds'.format(rate_type=cat)
            BV_SE = 'BV_{rate_type}_SE'.format(rate_type=cat)
            BY_SE = 'BY_{rate_type}_SE'.format(rate_type=cat)
            # 删除缺失值和值为0的行
            # df.dropna(subset=[BV_rate, BY_rate], inplace=True)
            df = df[~((df[BV_rate] == 0) | (df[BY_rate] == 0))].copy()
            df = df[(df['time_plot'] >= 2006) & (df['time_plot'] <= 2023)].copy()
            df.reset_index(inplace=True, drop=True)

            # 画图
            ax.plot(df['time_plot'], df[BV_rate], marker='o', markersize=2.3, color='#62B197',
                    linewidth=1.5, zorder=100,
                    label='B/Victoria', alpha=1)
            ax.plot(df['time_plot'], df[BY_rate], marker='o', markersize=2.3, color='#E18E6D',
                    linewidth=1.5, zorder=100,
                    label='B/Yamagata', alpha=1)

            # ax.errorbar(x=df['time_plot'], y=df[BV_rate], color='#62B197', linewidth=1.5,
            #             yerr=df[BV_SE], fmt='-', ecolor='#62B197', elinewidth=1, capsize=1.5, capthick=1,
            #             marker='o', ms=1.5, mew=1, label='B/Victoria', alpha=1)
            # ax.errorbar(df['time_plot'], df[BY_rate], color='#E18E6D', linewidth=1.5,
            #             yerr=df[BY_SE], fmt='-', ecolor='#E18E6D', elinewidth=1, capsize=1.5, capthick=1,
            #             marker='o', ms=1.5, mew=1, label='B/Yamagata', alpha=1)

            # 设置图例等
            # if cat == 'all':
            #     # ax.set_title("Global dN/dS".format(rate_type=cat.title()))
            # else:
            #     ax.set_title("{rate_type} dN/dS".format(rate_type=cat.title()), )
            ax.set_ylim(0, 0.3)
            ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
            ax.set_ylabel('dN/dS')
            ax.set_xlim(2006, 2023)
            ax.set_xticks(list(range(2006, 2024)))
            # ax.set_xticklabels([])
            ax.tick_params(axis='x', labelsize=6, rotation=90, pad=1)
            ax.tick_params(axis='y', labelsize=6, pad=1)
            # ax.axvline(x=2013, zorder=1, color='#bf3a2b', ls='--', linewidth='1.5')
            ax.grid(ls='-', zorder=0, alpha=0.5)
            ax.legend(ncol=2, frameon=False, loc='upper left', fontsize=6)

            ax.axvspan(2020+1/12, 2021+8/12, facecolor='#ef8c3b', edgecolor='none', alpha=.2)
            ax.set_title('{rate_type} dN/dS ratios '.format(rate_type=cat.title()), fontsize=7)
            ax.set_axisbelow(True)
            ax.set_xlabel('Year')

    # B
    cat = 'all'
    ax = fig.add_subplot(spec[0, 1])
    ax.text(-0.08, 0.95, 'B', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')

    # 指定画图的列
    BV_rate = 'BV_{rate_type}_dn/ds'.format(rate_type=cat)
    BY_rate = 'BY_{rate_type}_dn/ds'.format(rate_type=cat)
    BV_SE = 'BV_{rate_type}_SE'.format(rate_type=cat)
    BY_SE = 'BY_{rate_type}_SE'.format(rate_type=cat)
    # 删除缺失值和值为0的行
    # df.dropna(subset=[BV_rate, BY_rate], inplace=True)
    df = df[~((df[BV_rate] == 0) | (df[BY_rate] == 0))].copy()
    df = df[(df['time_plot'] >= 2006) & (df['time_plot'] <= 2023)].copy()
    df.reset_index(inplace=True, drop=True)

    # 画图
    ax.plot(df['time_plot'], df[BV_rate], marker='o', markersize=2.3, color='#62B197',
            linewidth=1.5, zorder=100,
            label='B/Victoria', alpha=1)
    ax.plot(df['time_plot'], df[BY_rate], marker='o', markersize=2.3, color='#E18E6D',
            linewidth=1.5, zorder=100,
            label='B/Yamagata', alpha=1)

    # ax.errorbar(x=df['time_plot'], y=df[BV_rate], color='#62B197', linewidth=1.5,
    #             yerr=df[BV_SE], fmt='-', ecolor='#62B197', elinewidth=1, capsize=1.5, capthick=1,
    #             marker='o', ms=1.5, mew=1, label='B/Victoria', alpha=1)
    # ax.errorbar(df['time_plot'], df[BY_rate], color='#E18E6D', linewidth=1.5,
    #             yerr=df[BY_SE], fmt='-', ecolor='#E18E6D', elinewidth=1, capsize=1.5, capthick=1,
    #             marker='o', ms=1.5, mew=1, label='B/Yamagata', alpha=1)

    # 设置图例等
    # if cat == 'all':
    #     # ax.set_title("Global dN/dS".format(rate_type=cat.title()))
    # else:
    #     ax.set_title("{rate_type} dN/dS".format(rate_type=cat.title()), )
    ax.set_ylim(0, 0.3)
    ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    ax.set_ylabel('dN/dS')
    ax.set_xlim(2006, 2023)
    ax.set_xticks(list(range(2006, 2024)))
    # ax.set_xticklabels([])
    ax.tick_params(axis='x', labelsize=6, rotation=90, pad=1)
    ax.tick_params(axis='y', labelsize=6, pad=1)
    # ax.axvline(x=2013, zorder=1, color='#bf3a2b', ls='--', linewidth='1.5')
    ax.grid(ls='-', zorder=0, alpha=0.5)
    ax.legend(ncol=2, frameon=False, loc='upper left', fontsize=6)
    ax.set_title('Global dN/dS ratios ', fontsize=7)
    ax.axvspan(2020 + 1 / 12, 2021 + 8 / 12, facecolor='#ef8c3b', edgecolor='none', alpha=.2)

    ax.set_axisbelow(True)
    if cat == 'internal':
        ax.set_xlabel('Year')

plt.savefig(r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BV_gasaid/BV_region_seq/data_process/final/figure/"
            r"paper_figure_v9/paper_fig4.pdf")
plt.show()





