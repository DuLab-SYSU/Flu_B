# -*- coding: utf-8 -*-
# @Time    : 2025/3/22 下午6:25
# @Author  : Hanwenjie
# @project : BY_dn_ds_cluster_MEME.py
# @File    : fig_s8.py
# @IDE     : PyCharm
# @REMARKS : description text
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
from scipy.interpolate import interp1d
from matplotlib.transforms import ScaledTranslation
mpl.rcParams['pdf.fonttype'] = 42


# import MCL percentage data
df_namerica = pd.read_csv(
    r"./result/BVNorthAmerica_cluster_percentage.csv", index_col=0)

for i in range(df_namerica.shape[0]):
    if df_namerica.loc[i, 'season'] == 'winter':
        df_namerica.loc[i, 'time_plot'] = df_namerica.loc[i, 'year'] + 1
    if df_namerica.loc[i, 'season'] == 'summer':
        df_namerica.loc[i, 'time_plot'] = df_namerica.loc[i, 'year'] + 0.5

df_asia = pd.read_csv(r"./result/BVAsia_cluster_percentage.csv",
                      index_col=0)

for i in range(df_asia.shape[0]):
    if df_asia.loc[i, 'season'] == 'winter':
        df_asia.loc[i, 'time_plot'] = df_asia.loc[i, 'year'] + 1
    if df_asia.loc[i, 'season'] == 'summer':
        df_asia.loc[i, 'time_plot'] = df_asia.loc[i, 'year'] + 0.5

# import MCL percentage data
df_europe = pd.read_csv(r"./result/BVEurope_cluster_percentage.csv",
                        index_col=0)

for i in range(df_europe.shape[0]):
    if df_europe.loc[i, 'season'] == 'winter':
        df_europe.loc[i, 'time_plot'] = df_europe.loc[i, 'year'] + 1
    if df_europe.loc[i, 'season'] == 'summer':
        df_europe.loc[i, 'time_plot'] = df_europe.loc[i, 'year'] + 0.5

df_samerica = pd.read_csv(
    r"./result/BVSouthAmerica_cluster_percentage.csv", index_col=0)

for i in range(df_samerica.shape[0]):
    if df_samerica.loc[i, 'season'] == 'winter':
        df_samerica.loc[i, 'time_plot'] = df_samerica.loc[i, 'year'] + 1
    if df_samerica.loc[i, 'season'] == 'summer':
        df_samerica.loc[i, 'time_plot'] = df_samerica.loc[i, 'year'] + 0.5

# import MCL percentage data
df_oceania = pd.read_csv(
    r"./result/BVOceania_cluster_percentage.csv", index_col=0)

for i in range(df_oceania.shape[0]):
    if df_oceania.loc[i, 'season'] == 'winter':
        df_oceania.loc[i, 'time_plot'] = df_oceania.loc[i, 'year'] + 1
    if df_oceania.loc[i, 'season'] == 'summer':
        df_oceania.loc[i, 'time_plot'] = df_oceania.loc[i, 'year'] + 0.5

df_africa = pd.read_csv(r"./result/BVAfrica_cluster_percentage.csv",
                        index_col=0)

for i in range(df_africa.shape[0]):
    if df_africa.loc[i, 'season'] == 'winter':
        df_africa.loc[i, 'time_plot'] = df_africa.loc[i, 'year'] + 1
    if df_africa.loc[i, 'season'] == 'summer':
        df_africa.loc[i, 'time_plot'] = df_africa.loc[i, 'year'] + 0.5

# specify unit conversion to cm
cm = 2.54

# plot
fig = plt.figure(figsize=(18.4 / cm, 18 / cm), dpi=300, layout="constrained")
spec = fig.add_gridspec(6, 1)
trans = ScaledTranslation(-30 / 300, 30 / 300, fig.dpi_scale_trans)
with mpl.rc_context(
        {'font.family': 'Arial', 'font.size': 7, 'hatch.linewidth': .5, 'lines.linewidth': .3, 'patch.linewidth': .3,
         'xtick.labelsize': 6, 'ytick.labelsize': 6}):
    # A panel
    ax = fig.add_subplot(spec[0, 0])
    ax.text(-0.04, 1.0, 'A', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    x = np.array(df_namerica['time_plot'])
    y1 = np.array(df_namerica['cluster1_proportion'])
    y2 = np.array(df_namerica['cluster2_proportion'])
    y3 = np.array(df_namerica['cluster3_proportion'])
    y4 = np.array(df_namerica['cluster4_proportion'])
    y5 = np.array(df_namerica['cluster5_proportion'])
    y6 = np.array(df_namerica['cluster6_proportion'])
    y7 = np.array(df_namerica['cluster7_proportion'])
   
    f_y1 = interp1d(x, y1, kind='cubic')
    f_y2 = interp1d(x, y2, kind='cubic')
    f_y3 = interp1d(x, y3, kind='cubic')
    f_y4 = interp1d(x, y4, kind='cubic')
    f_y5 = interp1d(x, y5, kind='cubic')
    f_y6 = interp1d(x, y6, kind='cubic')
    f_y7 = interp1d(x, y7, kind='cubic')


    x_interp = np.linspace(x.min(), x.max(), 10000)
    y1_interp = f_y1(x_interp)
    y2_interp = f_y2(x_interp)
    y3_interp = f_y3(x_interp)
    y4_interp = f_y4(x_interp)
    y5_interp = f_y5(x_interp)
    y6_interp = f_y6(x_interp)
    y7_interp = f_y7(x_interp)

    cluster_colors = ['#e67932', '#dcab3c', '#8cbb69', '#65ae96', '#5c6fd5', '#5d56cc', '#959797']
    cluster_labels = ['AU21', 'COV20', 'WA19', 'CO17', 'BR08', 'SD97', 'BJ87']
    ax.stackplot(x_interp, y2_interp, y6_interp, y3_interp, y5_interp, y1_interp, y4_interp, y7_interp, labels=cluster_labels, colors=cluster_colors, alpha=0.9)

    ax.text(x=2005, y=0.5, s='SD97', c='w', fontsize=7)
    ax.text(x=2013, y=0.5, s='BR08', c='w', fontsize=7)
    ax.text(x=2017.3, y=0.15, s='CO17', c='w', fontsize=7)
    ax.text(x=2020, y=0.5, s='WA19', c='w', fontsize=7)
    # ax.text(x=2020.3, y=0.5, s='COV20', c='w', fontsize=6, rotation=80)
    ax.text(x=2021.5, y=0.15, s='AU21', c='w', fontsize=7)

    ax.set_title("North America")
    ax.set_xlim(2003, 2023.5)
    ax.set_xticks(np.arange(2003, 2024, 1).tolist())
    ax.set_xticklabels([])
    ax.tick_params(axis='x', rotation=90, pad=1, labelsize=6, )
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_ylabel('Percentage')
    # ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08))

    # B panel
    ax = fig.add_subplot(spec[1, 0])
    ax.text(-0.04, 1.0, 'B', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    x = np.array(df_asia['time_plot'])
    y1 = np.array(df_asia['cluster1_proportion'])
    y2 = np.array(df_asia['cluster2_proportion'])
    y3 = np.array(df_asia['cluster3_proportion'])
    y4 = np.array(df_asia['cluster4_proportion'])
    y5 = np.array(df_asia['cluster5_proportion'])
    y6 = np.array(df_asia['cluster6_proportion'])
    y7 = np.array(df_asia['cluster7_proportion'])


    f_y1 = interp1d(x, y1, kind='cubic')
    f_y2 = interp1d(x, y2, kind='cubic')
    f_y3 = interp1d(x, y3, kind='cubic')
    f_y4 = interp1d(x, y4, kind='cubic')
    f_y5 = interp1d(x, y5, kind='cubic')
    f_y6 = interp1d(x, y6, kind='cubic')
    f_y7 = interp1d(x, y7, kind='cubic')


    x_interp = np.linspace(x.min(), x.max(), 10000)
    y1_interp = f_y1(x_interp)
    y2_interp = f_y2(x_interp)
    y3_interp = f_y3(x_interp)
    y4_interp = f_y4(x_interp)
    y5_interp = f_y5(x_interp)
    y6_interp = f_y6(x_interp)
    y7_interp = f_y7(x_interp)

    cluster_colors = ['#e67932', '#dcab3c', '#8cbb69', '#65ae96', '#5c6fd5', '#5d56cc', '#959797']
    cluster_labels = ['AU21', 'COV20', 'WA19', 'CO17', 'BR08', 'SD97', 'BJ87']
    ax.stackplot(x_interp, y2_interp, y6_interp, y3_interp, y5_interp, y1_interp, y4_interp, y7_interp, labels=cluster_labels, colors=cluster_colors, alpha=0.9)

    ax.text(x=2005, y=0.5, s='SD97', c='w', fontsize=7)
    ax.text(x=2013, y=0.5, s='BR08', c='w', fontsize=7)
    ax.text(x=2017.95, y=0.13, s='CO17', c='w', fontsize=7)
    ax.text(x=2019.2, y=0.5, s='WA19', c='w', fontsize=7)
    ax.text(x=2020.9, y=0.82, s='COV20', c='w', fontsize=6)
    ax.text(x=2021.5, y=0.15, s='AU21', c='w', fontsize=7)

    ax.set_title("Asia")
    ax.set_xlim(2003, 2023.5)
    ax.set_xticks(np.arange(2003, 2024, 1).tolist())
    ax.set_xticklabels([])
    ax.tick_params(axis='x', rotation=90, pad=1, labelsize=6, )
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_ylabel('Percentage')

    # C panel
    ax = fig.add_subplot(spec[2, 0])
    ax.text(-0.04, 1.0, 'C', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    x = np.array(df_europe['time_plot'])
    y1 = np.array(df_europe['cluster1_proportion'])
    y2 = np.array(df_europe['cluster2_proportion'])
    y3 = np.array(df_europe['cluster3_proportion'])
    y4 = np.array(df_europe['cluster4_proportion'])
    y5 = np.array(df_europe['cluster5_proportion'])
    y6 = np.array(df_europe['cluster6_proportion'])
    y7 = np.array(df_europe['cluster7_proportion'])

    f_y1 = interp1d(x, y1, kind='cubic')
    f_y2 = interp1d(x, y2, kind='cubic')
    f_y3 = interp1d(x, y3, kind='cubic')
    f_y4 = interp1d(x, y4, kind='cubic')
    f_y5 = interp1d(x, y5, kind='cubic')
    f_y6 = interp1d(x, y6, kind='cubic')
    f_y7 = interp1d(x, y7, kind='cubic')

    x_interp = np.linspace(x.min(), x.max(), 10000)
    y1_interp = f_y1(x_interp)
    y2_interp = f_y2(x_interp)
    y3_interp = f_y3(x_interp)
    y4_interp = f_y4(x_interp)
    y5_interp = f_y5(x_interp)
    y6_interp = f_y6(x_interp)
    y7_interp = f_y7(x_interp)

    cluster_colors = ['#e67932', '#dcab3c', '#8cbb69', '#65ae96', '#5c6fd5', '#5d56cc', '#959797']
    cluster_labels = ['AU21', 'COV20', 'WA19', 'CO17', 'BR08', 'SD97', 'BJ87']
    ax.stackplot(x_interp, y2_interp, y6_interp, y3_interp, y5_interp, y1_interp, y4_interp, y7_interp, labels=cluster_labels, colors=cluster_colors, alpha=0.9)

    ax.text(x=2005, y=0.5, s='SD97', c='w', fontsize=7)
    ax.text(x=2013, y=0.5, s='BR08', c='w', fontsize=7)
    ax.text(x=2017.6, y=0.15, s='CO17', c='w', fontsize=7)
    ax.text(x=2019.2, y=0.5, s='WA19', c='w', fontsize=7)
    ax.text(x=2019.2, y=0.0, s='COV20', c='w', fontsize=6)
    ax.text(x=2021.5, y=0.15, s='AU21', c='w', fontsize=7)

    ax.set_title("Europe")
    ax.set_xlim(2003, 2023.5)
    ax.set_xticks(np.arange(2003, 2024, 1).tolist())
    ax.set_xticklabels([])
    ax.tick_params(axis='x', rotation=90, pad=1, labelsize=6, )
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_ylabel('Percentage')
    # ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08))

    # D panel
    ax = fig.add_subplot(spec[3, 0])
    ax.text(-0.04, 1.0, 'D', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    x = np.array(df_samerica['time_plot'])
    y1 = np.array(df_samerica['cluster1_proportion'])
    y2 = np.array(df_samerica['cluster2_proportion'])
    y3 = np.array(df_samerica['cluster3_proportion'])
    y4 = np.array(df_samerica['cluster4_proportion'])
    y5 = np.array(df_samerica['cluster5_proportion'])
    y6 = np.array(df_samerica['cluster6_proportion'])
    # y7 = np.array(df_samerica['cluster7_proportion'])

    f_y1 = interp1d(x, y1, kind='cubic')
    f_y2 = interp1d(x, y2, kind='cubic')
    f_y3 = interp1d(x, y3, kind='cubic')
    f_y4 = interp1d(x, y4, kind='cubic')
    f_y5 = interp1d(x, y5, kind='cubic')
    f_y6 = interp1d(x, y6, kind='cubic')
    # f_y7 = interp1d(x, y7, kind='cubic')

    x_interp = np.linspace(x.min(), x.max(), 10000)
    y1_interp = f_y1(x_interp)
    y2_interp = f_y2(x_interp)
    y3_interp = f_y3(x_interp)
    y4_interp = f_y4(x_interp)
    y5_interp = f_y5(x_interp)
    y6_interp = f_y6(x_interp)
    # y7_interp = f_y7(x_interp)

    cluster_colors = ['#e67932', '#dcab3c', '#8cbb69', '#65ae96', '#5c6fd5', '#5d56cc', '#959797']
    cluster_labels = ['AU21', 'COV20', 'WA19', 'CO17', 'BR08', 'SD97', ]
    ax.stackplot(x_interp, y2_interp, y6_interp, y3_interp, y5_interp, y1_interp, y4_interp, labels=cluster_labels, colors=cluster_colors, alpha=0.9)

    ax.text(x=2005, y=0.5, s='SD97', c='w', fontsize=7)
    ax.text(x=2013, y=0.5, s='BR08', c='w', fontsize=7)
    ax.text(x=2018.2, y=0.5, s='CO17', c='w', fontsize=7)
    ax.text(x=2020.2, y=0.5, s='WA19', c='w', fontsize=7)
    ax.text(x=2019.9, y=0.0, s='COV20', c='w', fontsize=6)
    ax.text(x=2021.5, y=0.15, s='AU21', c='w', fontsize=7)

    ax.set_title("South America")
    ax.set_xlim(2003, 2023.5)
    ax.set_xticks(np.arange(2003, 2024, 1).tolist())
    ax.set_xticklabels([])
    ax.tick_params(axis='x', rotation=90, pad=1, labelsize=6, )
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_ylabel('Percentage')

    # E panel
    ax = fig.add_subplot(spec[4, 0])
    ax.text(-0.04, 1.0, 'E', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    x = np.array(df_oceania['time_plot'])
    y1 = np.array(df_oceania['cluster1_proportion'])
    y2 = np.array(df_oceania['cluster2_proportion'])
    y3 = np.array(df_oceania['cluster3_proportion'])
    y4 = np.array(df_oceania['cluster4_proportion'])
    y5 = np.array(df_oceania['cluster5_proportion'])
    y6 = np.array(df_oceania['cluster6_proportion'])
    y7 = np.array(df_oceania['cluster7_proportion'])

    f_y1 = interp1d(x, y1, kind='cubic')
    f_y2 = interp1d(x, y2, kind='cubic')
    f_y3 = interp1d(x, y3, kind='cubic')
    f_y4 = interp1d(x, y4, kind='cubic')
    f_y5 = interp1d(x, y5, kind='cubic')
    f_y6 = interp1d(x, y6, kind='cubic')
    f_y7 = interp1d(x, y7, kind='cubic')

    x_interp = np.linspace(x.min(), x.max(), 10000)
    y1_interp = f_y1(x_interp)
    y2_interp = f_y2(x_interp)
    y3_interp = f_y3(x_interp)
    y4_interp = f_y4(x_interp)
    y5_interp = f_y5(x_interp)
    y6_interp = f_y6(x_interp)
    y7_interp = f_y7(x_interp)

    cluster_colors = ['#e67932', '#dcab3c', '#8cbb69', '#65ae96', '#5c6fd5', '#5d56cc', '#959797']
    cluster_labels = ['AU21', 'COV20', 'WA19', 'CO17', 'BR08', 'SD97', 'BJ87']
    ax.stackplot(x_interp, y2_interp, y6_interp, y3_interp, y5_interp, y1_interp, y4_interp, y7_interp, labels=cluster_labels, colors=cluster_colors, alpha=0.9)

    ax.text(x=2005, y=0.5, s='SD97', c='w', fontsize=7)
    ax.text(x=2013, y=0.5, s='BR08', c='w', fontsize=7)
    ax.text(x=2018.4, y=0.6, s='CO17', c='w', fontsize=7)
    ax.text(x=2019.6, y=0.4, s='WA19', c='w', fontsize=7)
    ax.text(x=2019.4, y=0.0, s='COV20', c='w', fontsize=6)
    ax.text(x=2021.5, y=0.15, s='AU21', c='w', fontsize=7)

    ax.set_title("Oceania")
    ax.set_xlim(2003, 2023.5)
    ax.set_xticks(np.arange(2003, 2024, 1).tolist())
    ax.set_xticklabels([])
    ax.tick_params(axis='x', rotation=90, pad=1, labelsize=6, )
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_ylabel('Percentage')

    # F panel
    ax = fig.add_subplot(spec[5, 0])
    ax.text(-0.04, 1.0, 'F', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    x = np.array(df_africa['time_plot'])
    y1 = np.array(df_africa['cluster1_proportion'])
    y2 = np.array(df_africa['cluster2_proportion'])
    y3 = np.array(df_africa['cluster3_proportion'])
    y4 = np.array(df_africa['cluster4_proportion'])
    y5 = np.array(df_africa['cluster5_proportion'])
    y6 = np.array(df_africa['cluster6_proportion'])
    # y7 = np.array(df_africa['cluster7_proportion'])

    f_y1 = interp1d(x, y1, kind='cubic')
    f_y2 = interp1d(x, y2, kind='cubic')
    f_y3 = interp1d(x, y3, kind='cubic')
    f_y4 = interp1d(x, y4, kind='cubic')
    f_y5 = interp1d(x, y5, kind='cubic')
    f_y6 = interp1d(x, y6, kind='cubic')
    # f_y7 = interp1d(x, y7, kind='cubic')

    x_interp = np.linspace(x.min(), x.max(), 10000)
    y1_interp = f_y1(x_interp)
    y2_interp = f_y2(x_interp)
    y3_interp = f_y3(x_interp)
    y4_interp = f_y4(x_interp)
    y5_interp = f_y5(x_interp)
    y6_interp = f_y6(x_interp)
    # y7_interp = f_y7(x_interp)

    cluster_colors = ['#e67932', '#dcab3c', '#8cbb69', '#65ae96', '#5c6fd5', '#5d56cc', '#959797']
    cluster_labels = ['AU21', 'COV20', 'WA19', 'CO17', 'BR08', 'SD97', ]
    ax.stackplot(x_interp, y2_interp, y6_interp, y3_interp, y5_interp, y1_interp, y4_interp, labels=cluster_labels, colors=cluster_colors, alpha=0.9)

    ax.text(x=2005, y=0.5, s='SD97', c='w', fontsize=7)
    ax.text(x=2013, y=0.5, s='BR08', c='w', fontsize=7)
    ax.text(x=2018.5, y=0.8, s='CO17', c='w', fontsize=7)
    ax.text(x=2018.8, y=0.4, s='WA19', c='w', fontsize=7)
    ax.text(x=2018.6, y=0.0, s='COV20', c='w', fontsize=6)
    ax.text(x=2021.5, y=0.15, s='AU21', c='w', fontsize=7)

    ax.set_title("Africa")
    ax.set_xlim(2003, 2023.5)
    ax.set_xticks(np.arange(2003, 2024, 1).tolist())
    # ax.set_xticklabels(labels)
    ax.tick_params(axis='x', rotation=90, pad=1, labelsize=6, )
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Year')

plt.savefig(r"./figure/fig_s8.pdf")
plt.show()
