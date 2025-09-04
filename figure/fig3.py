# -*- coding: utf-8 -*-
# @Time    : 2025/3/22 17:42
# @Author  : Hanwenjie
# @project : BY_dn_ds_cluster_MEME.py
# @File    : fig3_v9.py
# @IDE     : PyCharm
# @REMARKS : Description text

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
mpl.rcParams['pdf.fonttype'] = 42


# Import MCL percentage data
df_north = pd.read_csv(r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BV_gasaid/BV_region_seq/data_process/"
                       r"final/MCL/nosample_update/log_MCL/cluster_percentage/BVNorth_cluster_percentage.csv", index_col=0)
# Set x-coordinate for plotting
for i in range(df_north.shape[0]):
    if df_north.loc[i,'season'] == 'winter':
        df_north.loc[i,'time_plot'] = df_north.loc[i,'year'] + 1
    if df_north.loc[i, 'season'] == 'summer':
        df_north.loc[i,'time_plot'] = df_north.loc[i,'year'] + 0.5

df_south = pd.read_csv(r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BV_gasaid/BV_region_seq/data_process/"
                       r"final/MCL/nosample_update/log_MCL/cluster_percentage/BVSouth_cluster_percentage.csv", index_col=0)
# Set x-coordinate for plotting
for i in range(df_south.shape[0]):
    if df_south.loc[i,'season'] == 'winter':
        df_south.loc[i,'time_plot'] = df_south.loc[i,'year'] + 1
    if df_south.loc[i, 'season'] == 'summer':
        df_south.loc[i,'time_plot'] = df_south.loc[i,'year'] + 0.5

# Specify unit conversion for centimeters
cm = 2.54

fig = plt.figure(figsize=(18.4/cm, 6.5/cm), dpi=300, layout="constrained")
spec = fig.add_gridspec(2,2)
trans = ScaledTranslation(-30/300, 30/300, fig.dpi_scale_trans)

with mpl.rc_context({'font.family': 'Arial', 'font.size': 7, 'hatch.linewidth': .5, 'lines.linewidth': .3, 'patch.linewidth': .3, 'xtick.labelsize': 6, 'ytick.labelsize': 6}):

    # Panel A
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
    # Set interpolation method to 'cubic' (default is 'zero')
    # Create interpolation function
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

    # Add cluster labels on plot
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
    # Set minor ticks
    ax.set_xticks(np.arange(1988, 2025, 2).tolist(), minor=True)  # Position of minor ticks
    ax.tick_params(axis='x', rotation=90, pad=1, labelsize=6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_ylabel('Percentage')

    # Panel B
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
    # Set interpolation method to 'cubic'
    # Create interpolation function
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
    # Add cluster labels on plot
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
    # Set minor ticks
    ax.set_xticks(np.arange(1988, 2025, 2).tolist(), minor=True)
    ax.tick_params(axis='x', rotation=90, pad=1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xlabel('Year')
    ax.set_ylabel('Percentage')

# (Similarly, the rest of the code for panels C and D can be translated following the same pattern)
