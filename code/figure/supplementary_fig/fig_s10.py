# -*- coding: utf-8 -*-
# @Time    : 2025/3/22 6:36 PM
# @Author  : Hanwenjie
# @project : BY_dn_ds_cluster_MEME.py
# @File    : fig_s10.py
# @IDE     : PyCharm
# @REMARKS : description text
import pandas as pd
import numpy as np
import baltic as bt
import dendropy
from io import StringIO
from tqdm import tqdm
from sklearn import preprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from matplotlib.transforms import ScaledTranslation
mpl.rcParams['pdf.fonttype']=42

# define a function to calculate entropy
def ent(data):
    prob1 = pd.value_counts(data) / len(data)
    return sum(np.log2(prob1) * prob1 * (-1))


# define a function to calculate information gain
def gain(data, str1, str2):
    e1 = data.groupby(str1).apply(lambda x: ent(x[str2]))
    p1 = pd.value_counts(data[str1]) / len(data[str1])
    e2 = sum(e1 * p1)
    return ent(data[str2]) - e2

def setAbsoluteTime(tree, dates):
    for i in tree.Objects:
        if i.is_leaf():
            i.absoluteTime=float(dates.loc[i.name, 'numeric date'])
        else:
            i.absoluteTime=float(dates.loc[i.traits['label'], 'numeric date'])
    tree.mostRecent = float(dates['numeric date'].max())
    return tree


# specify conversion unit for centimeters
cm = 2.54

# plot scatter of key sites
fig = plt.figure(figsize=(18.4/cm, 9.2/cm), dpi=300, layout="constrained")
spec = fig.add_gridspec(2,4)
trans = ScaledTranslation(-30/300, 30/300, fig.dpi_scale_trans)

BV_keysites = []
BY_keysites = []
with mpl.rc_context({'font.family': 'Arial', 'font.size': 7, 'hatch.linewidth': .5, 'lines.linewidth': .3, 'patch.linewidth': .3, 'xtick.labelsize': 6, 'ytick.labelsize': 6}):

    # BV cluster transition sites
    dict_df = {1: '7_4', 2: '4_1', 3: '1_5', 4: '1_3', 5: '3_6', 6: '6_2'}
    dict_cluster = {1: 'BJ87-SD97', 2: 'SD97-BR08', 3: 'BR08-CO17', 4: 'BR08-WA19', 5: 'WA19-COV20', 6: 'COV20-AU21', }
    fig_num = 'ABCDEF'
    num = 0
    for row in range(2):
        for col in range(3):
            num += 1
            df_gain = pd.read_csv(
                r'./result/BV_IG_{file}.csv'.format(file=dict_df[num]), index_col=0)

            # extract key site dataset
            df_key = df_gain.loc[(df_gain['entropy_norm'] > 0.3) & (df_gain['gain_norm'] > 0.4)].copy()
            df_key = df_key.sort_values(by='distance_top', ascending=True)
            df_key.reset_index(drop=True, inplace=True)
            df_key_top = df_key.loc[:2, :].copy()
            df_key_top.reset_index(inplace=True,drop=True)
            df_key_nontop = df_key.loc[3:,:].copy()
            df_key_nontop.reset_index(inplace=True, drop=True)

            # extract non-key site dataset
            df_other = df_gain.loc[(df_gain['entropy_norm'] <= 0.3) | (df_gain['gain_norm'] <= 0.4)].copy()
            df_other.reset_index(drop=True, inplace=True)
            print('{name}:'.format(name=dict_cluster[num]), df_key.loc[:, 'position'])
            BV_keysites = BV_keysites + list(df_key['position'])

            ax = fig.add_subplot(spec[row, col])
            ax.text(-0.25, 0.9, fig_num[num-1], transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
                    fontfamily='Arial')
            try:
                ax.scatter(x=df_key_nontop['gain_norm'], y=df_key_nontop['entropy_norm'], color='#a9373b',
                           s=70*df_key_nontop['distance_norm'], alpha=np.array(df_key_nontop['distance_norm']),zorder=100, clip_on=False)
            except ValueError:
                pass
            ax.scatter(x=df_key_top['gain_norm'], y=df_key_top['entropy_norm'], color='#a9373b', s=70*df_key_top['distance_norm'],
                       alpha=np.array(df_key_top['distance_norm']) , zorder=100, clip_on=False)

            # label positions of key sites
            for i in range(df_key_top.shape[0]):
                ax.text(df_key_top.loc[i, 'gain_norm'] - 0.01, df_key_top.loc[i, 'entropy_norm'] + 0.01, df_key_top.loc[i, 'position'],
                        color='#a9373b',fontsize=6, zorder=1000)

            # plot scatter of non-key sites
            ax.scatter(x=df_other['gain_norm'], y=df_other['entropy_norm'], color='#a9373b', s=70*df_other['distance_norm'],
                       alpha=np.array(df_other['distance_norm']), zorder=100, clip_on=False)

            ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.1, 0.2).tolist())
            ax.set_ylim(0, 1.18)
            ax.set_yticks(np.arange(0, 1.1, 0.2).tolist())

            ax.set_ylabel('Entropy')
            ax.set_xlabel('Information Gain')
            ax.set_title('{name}'.format(name=dict_cluster[num]), y=0.82,fontsize=7)
            # ax.axhline(y=0.3, zorder=0, color='black', ls='--', linewidth=1)
            # ax.axvline(x=0.4, ymax=1/1.18, zorder=0, color='black', ls='--', linewidth=1)
            ax.axhspan(1.0, 1.25, facecolor='grey', edgecolor='none', alpha=.3)
            ax.axhline(y=1.0, color='black', linewidth=1)
            # ax.set(xlim=(0, 1), ylim=(0, 1))
            ax.grid(axis='y', ls='--', zorder=0, alpha=0.5)
            # ax.grid(axis='x', ls='--', zorder=0, alpha=0.5)
            for x_value in np.arange(0.2, 1, 0.2):
                ax.axvline(x=x_value, ymax=1 / 1.18, zorder=0, color='grey', ls='--', linewidth=0.8,alpha=0.3)

    # BY cluster transition sites
    dict_df = {1: '3_2', 2: '2_1'}
    dict_cluster = {1: 'YA88-BJ93', 2: 'BJ93-WI10'}
    fig_num = 'GH'
    num = 0

    for row in [0,1]:
        for col in [3]:
            num += 1
            df_gain = pd.read_csv(
                r'./result/BY_IG_{file}.csv'.format(file=dict_df[num]), index_col=0)

            # extract key site dataset
            df_key = df_gain.loc[(df_gain['entropy_norm'] > 0.3) & (df_gain['gain_norm'] > 0.4)].copy()
            df_key = df_key.sort_values(by='distance_top', ascending=True)
            df_key.reset_index(drop=True, inplace=True)
            df_key_top = df_key.loc[:2, :].copy()
            df_key_top.reset_index(inplace=True, drop=True)
            df_key_nontop = df_key.loc[3:, :].copy()
            df_key_nontop.reset_index(inplace=True, drop=True)
            # extract non-key site dataset
            df_other = df_gain.loc[(df_gain['entropy_norm'] <= 0.3) | (df_gain['gain_norm'] <= 0.4)].copy()
            df_other.reset_index(drop=True, inplace=True)
            print('{name}:'.format(name=dict_cluster[num]), df_key.loc[:, 'position'])
            BY_keysites = BY_keysites + list(df_key['position'])

            ax = fig.add_subplot(spec[row, col])
            ax.text(-0.25, 0.9, fig_num[num - 1], transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
                    fontfamily='Arial')
            try:
                ax.scatter(x=df_key_nontop['gain_norm'], y=df_key_nontop['entropy_norm'], color='#a9373b',
                           s=70*df_key_nontop['distance_norm'], alpha=np.array(df_key_nontop['distance_norm']), zorder=100, clip_on=False)
            except ValueError:
                pass
            ax.scatter(x=df_key_top['gain_norm'], y=df_key_top['entropy_norm'], color='#a9373b',
                       s=70*df_key_top['distance_norm'], alpha=np.array(df_key_top['distance_norm']), zorder=100, clip_on=False)

            # label positions of key sites
            for i in range(df_key_top.shape[0]):
                ax.text(df_key_top.loc[i, 'gain_norm'] - 0.01, df_key_top.loc[i, 'entropy_norm'] + 0.01,
                        df_key_top.loc[i, 'position'],
                        color='#a9373b', fontsize=6, zorder=1000)

            # plot scatter of non-key sites
            ax.scatter(x=df_other['gain_norm'], y=df_other['entropy_norm'], color='#a9373b',
                       s=70*df_other['distance_norm'], alpha=np.array(df_other['distance_norm']), zorder=100, clip_on=False)

            ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.1, 0.2).tolist())
            ax.set_ylim(0, 1.18)
            ax.set_yticks(np.arange(0, 1.1, 0.2).tolist())

            ax.set_ylabel('Entropy')
            ax.set_xlabel('Information Gain')
            ax.set_title('{name}'.format(name=dict_cluster[num]), fontsize=7,y=0.82)
            # ax.axhline(y=0.3, zorder=0, color='black', ls='--', linewidth=1)
            # ax.axvline(x=0.4, ymax=1/1.18, zorder=0, color='black', ls='--', linewidth=1)
            ax.axhspan(1.0, 1.25, facecolor='grey', edgecolor='none', alpha=.3)
            ax.axhline(y=1.0, color='black', linewidth=1)
            # ax.set(xlim=(0, 1), ylim=(0, 1))
            ax.grid(axis='y', ls='--', zorder=0, alpha=0.5)
            # ax.grid(axis='x', ls='--', zorder=0, alpha=0.5)
            for x_value in np.arange(0.2, 1, 0.2):
                ax.axvline(x=x_value, ymax=1 / 1.18, zorder=0, color='grey', ls='--', linewidth=0.8,alpha=0.3)

plt.savefig(r"./figure/fig_s10.pdf")
plt.show()
