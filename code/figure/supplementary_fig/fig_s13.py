# -*- coding: utf-8 -*-
# @Time    : 2025/4/3 4:54 PM
# @Author  : Hanwenjie
# @project : BY_dn_ds_cluster_MEME.py
# @File    : fig_s13.py
# @IDE     : PyCharm
# @REMARKS : Description text
import pandas as pd
import baltic as bt
import dendropy
from io import StringIO
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from matplotlib.transforms import ScaledTranslation
mpl.rcParams['pdf.fonttype']=42

# Specify unit conversion for centimeters
cm = 2.54

fig = plt.figure(figsize=(12.1/cm, 8/cm), dpi=300, layout="constrained")
spec = fig.add_gridspec(2,1)
trans = ScaledTranslation(-30/300, 30/300, fig.dpi_scale_trans)
with mpl.rc_context({'font.family': 'Arial', 'font.size': 7, 'hatch.linewidth': .5, 'lines.linewidth': .3, 'patch.linewidth': .3, 'xtick.labelsize': 6, 'ytick.labelsize': 6}):

    # i panel
    dict_df_BV = {1: '7_4', 2: '4_1', 3: '1_5', 4: '1_3', 5: '3_6', 6: '6_2'}
    dict_cluster_BV = {1: 'BJ87-SD97', 2: 'SD97-BR08', 3: 'BR08-CO17', 4: 'BR08-WA19', 5: 'WA19-COV20', 6: 'COV20-AU21'}
    # Select different numbers of top sites based on distance
    for num_sites in [2]:
        list_A = []
        list_B = []
        list_C = []
        list_D = []
        list_E = []

        ax = fig.add_subplot(spec[0, 0])
        ax.text(-0.05, 0.9, 'A', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
                fontfamily='Arial')

        BV_trans = ['BJ87-SD97', 'SD97-BR08', 'BR08-CO17', 'BR08-WA19', 'WA19-COV20', 'COV20-AU21']

        for num in range(1, len(dict_df_BV) + 1):
            df_gain = pd.read_csv(
                r'./result/BV_IG_{file}.csv'.format(file=dict_df_BV[num]), index_col=0)
            # Extract dataset of key sites
            df_key = df_gain.loc[(df_gain['entropy_norm'] > 0.3) & (df_gain['gain_norm'] > 0.4)].copy()
            df_key = df_key.sort_values(by='distance_norm', ascending=False)
            df_key.reset_index(drop=True, inplace=True)
            print(dict_cluster_BV[num], ':')
            print(df_key)

            df_key_top = df_key.loc[:num_sites, :].copy()
            # df_key_top = df_key.copy()
            print(df_key_top)
            df_epitope_kind = df_key_top['epitope'].value_counts()
            for epitope in ['A', 'B', 'C', 'D', 'E']:
                if epitope not in df_epitope_kind.index:
                    df_epitope_kind[epitope] = 0

            list_A.append(df_epitope_kind['A'])
            list_B.append(df_epitope_kind['B'])
            list_C.append(df_epitope_kind['C'])
            list_D.append(df_epitope_kind['D'])
            list_E.append(df_epitope_kind['E'])

            print(df_epitope_kind.to_dict())
            print(df_epitope_kind.index)
            print('\n')

        BY_trans = ['YA88-BJ93', 'BJ93-WI10']
        dict_df_BY = {1: '3_2', 2: '2_1'}
        dict_cluster_By = {1: 'YA88-BJ93', 2: 'BJ93-WI10'}

        for num in range(1, len(dict_df_BY) + 1):
            df_gain = pd.read_csv(
                r'./result/BY_IG_{file}.csv'.format(file=dict_df_BY[num]), index_col=0)
            # Extract dataset of key sites
            df_key = df_gain.loc[(df_gain['entropy_norm'] > 0.3) & (df_gain['gain_norm'] > 0.4)].copy()
            df_key = df_key.sort_values(by='distance_norm', ascending=False)
            df_key.reset_index(drop=True, inplace=True)
            # print(dict_cluster[num], ':')
            # print(df_key)

            df_key_top = df_key.loc[:num_sites, :].copy()
            # df_key_top = df_key.copy()
            print(df_key_top)
            df_epitope_kind = df_key_top['epitope'].value_counts()
            for epitope in ['A', 'B', 'C', 'D', 'E']:
                if epitope not in df_epitope_kind.index:
                    df_epitope_kind[epitope] = 0

            list_A.append(df_epitope_kind['A'])
            list_B.append(df_epitope_kind['B'])
            list_C.append(df_epitope_kind['C'])
            list_D.append(df_epitope_kind['D'])
            list_E.append(df_epitope_kind['E'])

        site_distribution = {'A': list_A, 'B': list_B, 'C': list_C, 'D': list_D, 'E': list_E}
        # colors = ['#bf3a2b', '#3e58cf', '#4b8ec1', '#65ae96', '#8cbb69',]
        colors = ['#F1AA4E', '#65B779', '#9081A7', '#E958A1', '#C8C8C8', ]
        print(site_distribution)

        x = np.arange(len(BV_trans) + len(BY_trans))  # the label locations
        width = 0.15  # the width of the bars
        multiplier = 0

        i = 0
        for attribute, measurement in site_distribution.items():
            offset = width * multiplier
            bars = ax.bar(x + offset, measurement, width, label=attribute, color=colors[i], alpha=0.850, zorder=5)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')

            multiplier += 1
            i += 1
        ax.set_xlim(-0.35, 8)
        x_ticks = [x_tick for x_tick in (x + 2 * width)]
        print(x_ticks)
        # x_ticks = [x for pair in zip(x, x + 2*width) for x in pair]
        x_labels = [x_label for x_label in (BV_trans + BY_trans)]
        ax.set_xticks(x_ticks, x_labels)
        ax.tick_params(axis='x', rotation=30, pad=1)
        ax.set_ylim(0, 4.2)
        ax.set_yticks(np.arange(0, 3.1, 1))
        ax.set_xlabel('')
        ax.set_ylabel('No. of key sites')

        # ax.set_title('Distribution of key sites in epitopes',pad=11)
        ax.grid(axis='y', ls='--', alpha=0.5, zorder=0)
        ax.spines['bottom'].set_zorder(10)
        ax.legend(['epi-A', 'epi-B', 'epi-C', 'epi-D', 'epi-E'], frameon=False, ncols=5, fontsize=7,
                  loc='lower left', bbox_to_anchor=(0, 0.94), columnspacing=.8, handletextpad=0.35, handlelength=2)
        ax.axvline(x=5.8, zorder=0, color='black', ls='-', linewidth=1)
        ax.axhspan(3.6, 4.2, facecolor='grey', edgecolor='none', alpha=.3)
        ax.axhline(y=3.6, color='black', linewidth=1)
        ax.text(2.2, 3.75, 'B/Victoria')
        ax.text(6.12, 3.75, 'B/Yamagata')

        # j panel
        dict_df_BV = {1: '7_4', 2: '4_1', 3: '1_5', 4: '1_3', 5: '3_6', 6: '6_2'}
        dict_cluster_BV = {1: 'BJ87-SD97', 2: 'SD97-BR08', 3: 'BR08-CO17', 4: 'BR08-WA19', 5: 'WA19-COV20',
                           6: 'COV20-AU21'}
        # Select different numbers of top sites based on distance
        for num_sites in [2]:
            list_rbs = []
            list_norbs = []

            ax = fig.add_subplot(spec[1, 0])
            ax.text(-0.05, 0.9, 'B', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
                    fontfamily='Arial')

            BV_trans = ['BJ87-SD97', 'SD97-BR08', 'BR08-CO17', 'BR08-WA19', 'WA19-COV20', 'COV20-AU21']

            for num in range(1, len(dict_df_BV) + 1):
                df_gain = pd.read_csv(
                    r'./result/BV_IG_{file}.csv'.format(file=dict_df_BV[num]), index_col=0)
                # Extract dataset of key sites
                df_key = df_gain.loc[(df_gain['entropy_norm'] > 0.3) & (df_gain['gain_norm'] > 0.4)].copy()
                df_key = df_key.sort_values(by='distance_norm', ascending=False)
                df_key.reset_index(drop=True, inplace=True)
                print(dict_cluster_BV[num], ':')
                print(df_key)

                df_key_top = df_key.loc[:num_sites, :].copy()
                # df_key_top = df_key.copy()
                print(df_key_top)
                df_epitope_kind = df_key_top['rbs'].value_counts()
                for epitope in ['rbs', 'no_rbs']:
                    if epitope not in df_epitope_kind.index:
                        df_epitope_kind[epitope] = 0
                print(df_epitope_kind)

                list_rbs.append(df_epitope_kind['rbs'])
                list_norbs.append(df_epitope_kind['no_rbs'])

                # print(df_epitope_kind.to_dict())
                # print(df_epitope_kind.index)
                print('\n')

            BY_trans = ['YA88-BJ93', 'BJ93-WI10']
            dict_df_BY = {1: '3_2', 2: '2_1'}
            dict_cluster_By = {1: 'YA88-BJ93', 2: 'BJ93-WI10'}

            for num in range(1, len(dict_df_BY) + 1):
                df_gain = pd.read_csv(
                    r'./result/BY_IG_{file}.csv'.format(file=dict_df_BY[num]), index_col=0)
                # Extract dataset of key sites
                df_key = df_gain.loc[(df_gain['entropy_norm'] > 0.3) & (df_gain['gain_norm'] > 0.4)].copy()
                df_key = df_key.sort_values(by='distance_norm', ascending=False)
                df_key.reset_index(drop=True, inplace=True)
                # print(dict_cluster[num], ':')
                # print(df_key)

                df_key_top = df_key.loc[:num_sites, :].copy()
                # df_key_top = df_key.copy()
                print(df_key_top)
                df_epitope_kind = df_key_top['rbs'].value_counts()
                for epitope in ['rbs', 'no_rbs']:
                    if epitope not in df_epitope_kind.index:
                        df_epitope_kind[epitope] = 0
                print(df_epitope_kind)

                list_rbs.append(df_epitope_kind['rbs'])
                list_norbs.append(df_epitope_kind['no_rbs'])

            site_distribution = {'RBS': list_rbs, 'NON-RBS': list_norbs}
            colors = ['#fe817d', '#81b8df']
            print(site_distribution)

            x = np.arange(len(BV_trans) + len(BY_trans))  # the label locations
            width = 0.15  # the width of the bars
            multiplier = 0

            i = 0
            for attribute, measurement in site_distribution.items():
                offset = width * multiplier
                bars = ax.bar(x + offset, measurement, width, label=attribute, color=colors[i], alpha=0.850, zorder=5)
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')

                multiplier += 1
                i += 1
            ax.set_xlim(-0.35, 7.65)
            x_ticks = [x_tick for x_tick in (x + 0.5 * width)]
            print(x_ticks)
            # x_ticks = [x for pair in zip(x, x + 2*width) for x in pair]
            x_labels = [x_label for x_label in (BV_trans + BY_trans)]
            ax.set_xticks(x_ticks, x_labels)
            ax.tick_params(axis='x', rotation=30, pad=1)
            ax.set_ylim(0, 4.2)
            ax.set_yticks(np.arange(0, 3.1, 1))
            ax.set_xlabel('')
            ax.set_ylabel('No. of key sites')

            # ax.set_title('Distribution of key sites in RBS', pad=11)
            ax.grid(axis='y', ls='--', alpha=0.5, zorder=0)
            ax.spines['bottom'].set_zorder(10)
            ax.legend(['RBS', 'NON-RBS'], frameon=False, ncols=5, fontsize=7,
                      loc='lower left', bbox_to_anchor=(0.25, 0.94), columnspacing=.8, handletextpad=0.35,
                      handlelength=2)
            ax.axvline(x=5.8, zorder=0, color='black', ls='-', linewidth=1)
            ax.axhspan(3.6, 4.2, facecolor='grey', edgecolor='none', alpha=.3)
            ax.axhline(y=3.6, color='black', linewidth=1)
            ax.text(2.2, 3.75, 'B/Victoria')
            ax.text(6.02, 3.75, 'B/Yamagata')

plt.savefig(r"./figure/fig_s13.pdf")
plt.show()
