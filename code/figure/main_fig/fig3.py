# -*- coding: utf-8 -*-
# @Time    : 2025/3/22 下午5:42
# @Author  : Hanwenjie
# @project : BY_dn_ds_cluster_MEME.py
# @File    : fig3.py
# @IDE     : PyCharm
# @REMARKS : description text
import pandas as pd
import baltic as bt
import dendropy
import seaborn as sns
from io import StringIO
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from scipy.interpolate import interp1d
from matplotlib.transforms import ScaledTranslation
mpl.rcParams['pdf.fonttype']=42


#import MCL percentage data
df_north = pd.read_csv(r"./result/BVNorth_cluster_percentage.csv",index_col=0)
# Set the x-coordinate
for i in range(df_north.shape[0]):
    if df_north.loc[i,'season'] == 'winter':
        df_north.loc[i,'time_plot'] = df_north.loc[i,'year'] + 1
    if df_north.loc[i, 'season'] == 'summer':
        df_north.loc[i,'time_plot'] = df_north.loc[i,'year'] + 0.5

df_south = pd.read_csv(r"./result/BVSouth_cluster_percentage.csv",index_col=0)
# Set the x-coordinate
for i in range(df_south.shape[0]):
    if df_south.loc[i,'season'] == 'winter':
        df_south.loc[i,'time_plot'] = df_south.loc[i,'year'] + 1
    if df_south.loc[i, 'season'] == 'summer':
        df_south.loc[i,'time_plot'] = df_south.loc[i,'year'] + 0.5

# specify unit conversion to cm
cm = 2.54

fig = plt.figure(figsize=(18.4/cm, 6.5/cm), dpi=300, layout="constrained")
spec = fig.add_gridspec(2,2)
trans = ScaledTranslation(-30/300, 30/300, fig.dpi_scale_trans)
with mpl.rc_context({'font.family': 'Arial', 'font.size': 7, 'hatch.linewidth': .5, 'lines.linewidth': .3, 'patch.linewidth': .3, 'xtick.labelsize': 6, 'ytick.labelsize': 6}):

    # A panel
    ax = fig.add_subplot(spec[0, 0])
    ax.text(-0.1, 0.93, 'A', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom', fontfamily='Arial')
    x = np.array(df_north['time_plot'])
    y1 = np.array(df_north['cluster1_proportion'])
    y2 = np.array(df_north['cluster2_proportion'])
    y3 = np.array(df_north['cluster3_proportion'])
    y4 = np.array(df_north['cluster4_proportion'])
    y5 = np.array(df_north['cluster5_proportion'])
    y6 = np.array(df_north['cluster6_proportion'])
    y7 = np.array(df_north['cluster7_proportion'])
    # Set the interpolation method to 'linear', with the default being 'zero'
    # Create Interpolation functions
    f_y1 = interp1d(x, y1, kind='cubic')
    f_y2 = interp1d(x, y2, kind='cubic')
    f_y3 = interp1d(x, y3, kind='cubic')
    f_y4 = interp1d(x, y4, kind='cubic')
    f_y5 = interp1d(x, y5, kind='cubic')
    f_y6 = interp1d(x, y6, kind='cubic')
    f_y7 = interp1d(x, y7, kind='cubic')

    # Generate interpolated data
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

    ax.text(x=1987.6, y=0.7, s='BJ87', c='w', fontsize=7)
    ax.text(x=1997, y=0.5, s='SD97', c='w', fontsize=7)
    ax.text(x=2011.5, y=0.5, s='BR08', c='w', fontsize=7)
    ax.text(x=2016.8, y=0.1, s='CO17', c='w', fontsize=5, rotation=0)
    ax.text(x=2018.7, y=0.5, s='WA19', c='w', fontsize=5, rotation=0)
    ax.text(x=2020.8, y=0.85, s='COV20', c='w', fontsize=5, rotation=0)
    ax.text(x=2021, y=0.1, s='AU21', c='w', fontsize=6, rotation=0)
    ax.set_title("B/Victoria-Northern Hemisphere", fontsize=7)
    ax.set_xlim(1987, 2024)
    ax.set_xticks(np.arange(1987, 2024, 2).tolist())
    ax.set_xticklabels([])
    # Set the secondary scale
    ax.set_xticks(np.arange(1988, 2025, 2).tolist(), minor=True) 
    # ax.grid(which='minor', alpha=0.2) 
    ax.tick_params(axis='x', rotation=90, pad=1, labelsize=6, )
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_ylabel('Percentage')
    # ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08))

    # B panel
    ax = fig.add_subplot(spec[1, 0])
    ax.text(-0.1, 0.93, '', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom', fontfamily='Arial')
    x = np.array(df_south['time_plot'])
    y1 = np.array(df_south['cluster1_proportion'])
    y2 = np.array(df_south['cluster2_proportion'])
    y3 = np.array(df_south['cluster3_proportion'])
    y4 = np.array(df_south['cluster4_proportion'])
    y5 = np.array(df_south['cluster5_proportion'])
    y6 = np.array(df_south['cluster6_proportion'])
    y7 = np.array(df_south['cluster7_proportion'])

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
    # sns.kdeplot(data=)
    ax.text(x=1989.5, y=0.7, s='BJ87', c='w', fontsize=7)
    ax.text(x=2001, y=0.5, s='SD97', c='w', fontsize=7)
    ax.text(x=2011.5, y=0.5, s='BR08', c='w', fontsize=7)
    ax.text(x=2018.3, y=0.85, s='CO17', c='w', fontsize=5, rotation=0)
    ax.text(x=2018.5, y=0.4, s='WA19', c='w', fontsize=5, rotation=0)
    ax.text(x=2018.6, y=0.0, s='COV20', c='w', fontsize=5, rotation=0)
    ax.text(x=2020.8, y=0.3, s='AU21', c='w', fontsize=6, rotation=0)
    ax.set_title("B/Victoria-Southern Hemisphere", fontsize=7)
    ax.set_xlim(1987, 2024)
    ax.set_xticks((np.arange(1987, 2024, 2).tolist()))
    ax.set_xticklabels(np.arange(1987, 2024, 2).tolist())

    ax.set_xticks(np.arange(1988, 2025, 2).tolist(), minor=True) 
    # ax.grid(which='minor', alpha=0.2)  

    ax.tick_params(axis='x', rotation=90, pad=1,)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xlabel('Year')
    ax.set_ylabel('Percentage')

    # import MCL percentage data
    df_north = pd.read_csv(
        r"./result/BYNorth_cluster_percentage.csv", index_col=0)

    for i in range(df_north.shape[0]):
        if df_north.loc[i, 'season'] == 'winter':
            df_north.loc[i, 'time_plot'] = df_north.loc[i, 'year'] + 1
        if df_north.loc[i, 'season'] == 'summer':
            df_north.loc[i, 'time_plot'] = df_north.loc[i, 'year'] + 0.5

    df_south = pd.read_csv(
        r"./result/BYSouth_cluster_percentage.csv", index_col=0)
    for i in range(df_south.shape[0]):
        if df_south.loc[i, 'season'] == 'winter':
            df_south.loc[i, 'time_plot'] = df_south.loc[i, 'year'] + 1
        if df_south.loc[i, 'season'] == 'summer':
            df_south.loc[i, 'time_plot'] = df_south.loc[i, 'year'] + 0.5

    # C panel
    ax = fig.add_subplot(spec[0, 1])
    ax.text(-0.06, 0.93, 'B', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom', fontfamily='Arial')
    x = np.array(df_north['time_plot'])
    y1 = np.array(df_north['cluster1_proportion']+df_north['cluster4_proportion'])
    y2 = np.array(df_north['cluster2_proportion'])
    y3 = np.array(df_north['cluster3_proportion'])

    f_y1 = interp1d(x, y1, kind='cubic')
    f_y2 = interp1d(x, y2, kind='cubic')
    f_y3 = interp1d(x, y3, kind='cubic')

    x_interp = np.linspace(x.min(), x.max(), 10000)
    y1_interp = f_y1(x_interp)
    y2_interp = f_y2(x_interp)
    y3_interp = f_y3(x_interp)

    # cluster_colors = ['#54a556','#ef8c3b','#477fb8']
    cluster_colors = ['#cbb742', '#7eb876', '#4988c5']
    # cluster_labels = ['BR08', 'WA19', 'AU21', 'SD97', 'CO17', 'COV20','BJ87']
    # cluster_labels = ['AU21', 'COV20', 'WA19', 'CO17', 'BR08', 'SD97', 'BJ87']
    ax.stackplot(x_interp, y1_interp, y2_interp, y3_interp, colors=cluster_colors, alpha=0.9)

    ax.text(x=1989, y=0.5, s='YA88', c='w', fontsize=7)
    ax.text(x=2000, y=0.5, s='BJ93', c='w', fontsize=7)
    ax.text(x=2011, y=0.06, s='WI10', c='w', fontsize=7)
    ax.set_title("B/Yamagata-Northern Hemisphere", fontsize=7)
    # ax.set_xlim(1988, 2020)
    # ax.set_xtickscks(np.arange(1988, 2021, 2).tolist())
    ax.set_xlim(1988, 2024)
    ax.set_xticks(np.arange(1988, 2024, 2).tolist())
    ax.set_xticklabels([])

    ax.set_xticks(np.arange(1989, 2024, 2).tolist(), minor=True)
    # ax.tick_params(axis='x')
    ax.tick_params(axis='x', rotation=90, pad=1, )
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.axvspan(2020, 2024, ymax=1, hatch='//', facecolor='none', edgecolor='#c3c3c3', linewidth=0.1)
    # ax.set_ylabel('Percentage')
    # ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.08))

    # D panel
    ax = fig.add_subplot(spec[1, 1])
    ax.text(-0.1, 0.93, '', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom', fontfamily='Arial')
    x = np.array(df_south['time_plot'])
    y1 = np.array(df_south['cluster1_proportion'])
    y2 = np.array(df_south['cluster2_proportion'])
    y3 = np.array(df_south['cluster3_proportion'])

    f_y1 = interp1d(x, y1, kind='cubic')
    f_y2 = interp1d(x, y2, kind='cubic')
    f_y3 = interp1d(x, y3, kind='cubic')

    x_interp = np.linspace(x.min(), x.max(), 10000)
    y1_interp = f_y1(x_interp)
    y2_interp = f_y2(x_interp)
    y3_interp = f_y3(x_interp)
    cluster_colors = ['#cbb742', '#7eb876', '#4988c5']
    ax.stackplot(x_interp, y1_interp, y2_interp, y3_interp, colors=cluster_colors, alpha=0.9)

    ax.text(x=1991.2, y=0.8, s='YA88', c='w', fontsize=7)
    ax.text(x=2000, y=0.5, s='BJ93', c='w', fontsize=7)
    ax.text(x=2011, y=0.06, s='WI10', c='w', fontsize=7)
    ax.set_title("B/Yamagata-Southern hemisphere", fontsize=7)
    # ax.set_xlim(1988, 2020)
    # ax.set_xticks(np.arange(1988, 2021, 2).tolist())
    ax.set_xlim(1988, 2024)
    ax.set_xticks(np.arange(1988, 2024, 2).tolist())

    ax.set_xticks(np.arange(1989, 2024, 2).tolist(), minor=True) 
    # ax.set_xticklabels(labels)
    ax.tick_params(axis='x', rotation=90, pad=1,)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # ax.set_ylabel('Percentage')
    ax.set_xlabel('Year')
    ax.axvspan(1988, 1991, ymax=1, hatch='//', facecolor='none', edgecolor='#c3c3c3', linewidth=0.1)
    ax.axvspan(2020, 2024, ymax=1, hatch='//', facecolor='none', edgecolor='#c3c3c3', linewidth=0.1)

    # ax.add_patch(patches.Rectangle((1988,0), 1991-1988, 1, hatch='//', color='grey'))

plt.savefig(r"./figure/fig3.pdf")
plt.show()
