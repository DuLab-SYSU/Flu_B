# -*- coding: utf-8 -*-
# @Time    : 2025/3/31 9:03 PM
# @Author  : Hanwenjie
# @project : BY_dn_ds_cluster_MEME.py
# @File    : fig1.py
# @IDE     : PyCharm
# @REMARKS : Description text
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.transforms import ScaledTranslation
from matplotlib.ticker import AutoMinorLocator, AutoLocator
mpl.rcParams['pdf.fonttype'] = 42
import geopandas as gpd
import geopandas
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np
import baltic as bt
import dendropy
from io import StringIO
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from matplotlib.transforms import ScaledTranslation
import markov_clustering as mc
import networkx as nx
import random
from tqdm import tqdm
from multiprocessing import Pool
from datetime import datetime
from matplotlib.patches import Wedge
from scipy.interpolate import interp1d
mpl.rcParams['pdf.fonttype'] = 42
import math

def date_to_decimal(date):
    # Define start date
    start_date = datetime(date.year, 1, 1)

    # Calculate the difference between dates, convert to days
    days_diff = (date - start_date).days

    # Get the start year
    start_year = start_date.year

    # Calculate the number of days in a year
    days_in_year = 365 if (start_year % 4 == 0 and (start_year % 100 != 0 or start_year % 400 == 0)) else 366

    # Calculate the proportion of the date within the year
    year_ratio = days_diff / days_in_year

    # Convert the date to a decimal representation of the year
    date_as_decimal = start_year + year_ratio

    return date_as_decimal


def drawPieMarker(ax, x, y, ratios, sizes, colors):
    assert sum(ratios) <= 1, 'sum of ratios needs to be <= 1'

    markers = []
    previous = 0
    for color, ratio in zip(colors, ratios):
        this = 360 * ratio + previous
        marker = Wedge(center=(x, y), r=sizes, theta1=previous, theta2=this, facecolor=color, edgecolor='k', linewidth=0.15)
        markers.append(marker)
        previous = this

    for marker in markers:
        ax.add_patch(marker)

    return ax
def Geoplot(ax):
    ax.axis('off')
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    world = world[(world.name != "Antarctica") & (world.name != "Fr. S. Antarctic Lands")]
    myplot = world.plot(
        ax=ax,
        linewidth=0.3,
        color='lightgrey', edgecolor='white',
        missing_kwds={"color": "lightgrey", "edgecolor": "white", "label": "Missing values"})

    rat = 1

    world = world.to_crs('EPSG:4326')
    for name, centroid in zip(world.name, world.centroid):

        if name in country_dct:
            num_lst = country_dct[name]
            ratio_lst = [i / sum(num_lst) for i in num_lst]
            if sum(ratio_lst) > 1:
                ratio_lst = [int(i / sum(num_lst) * 100) / 100 for i in num_lst]
            drawPieMarker(ax,
                          x=centroid.x,
                          y=centroid.y,
                          ratios=ratio_lst,
                          sizes=[np.log(sum(num_lst)) / rat],
                          colors=[color_dct[i] for i in variant_lst])

    return ax

def date_num(date):
    date_number = int(date[:4]) + (int(date[-2:]) - 1) / 12
    return date_number

# import case data
df_case = pd.read_csv(
    r"./data/BV_BY_case.csv", index_col=0)
df_case['date'] = pd.to_datetime(df_case['date'])
df_case = df_case[df_case['date'] >= '2015-10-1'].copy()
df_case.reset_index(inplace=True, drop=True)
for i in range(df_case.shape[0]):
    df_case.loc[i, 'time'] = df_case.index[i] * (102 / 438)

print(df_case)

# import seq count data
df_BV_seq = pd.read_csv(
    r"./data/BV_seq_count.csv", index_col=0)
df_BY_seq = pd.read_csv(
    r"./data/BY_seq_count.csv", index_col=0)
df_BV_seq = df_BV_seq[df_BV_seq['year_month'] >= '2015-10'].copy()
df_BV_seq.reset_index(inplace=True, drop=True)
df_BY_seq = df_BY_seq[df_BY_seq['year_month'] >= '2015-10'].copy()
df_BY_seq.reset_index(inplace=True, drop=True)

df_seq_count = pd.merge(df_BV_seq, df_BY_seq, on='year_month', how='outer')
df_seq_count.reset_index(inplace=True, drop=True)
df_seq_count = df_seq_count.fillna(0)
print(df_seq_count)


# Specify conversion unit for cm
cm = 2.54
# Plotting
fig = plt.figure(figsize=(18.4/ cm, 12/ cm), dpi=300, layout="compressed")
# spec = fig.add_gridspec(5, 2, width_ratios=[1,1,1], height_ratios=[2, 2, 2])
# spec = fig.add_gridspec(3, 4)
spec = fig.add_gridspec(4, 4, height_ratios=[14/3,14/3,14/3,1])
trans = ScaledTranslation(-30 / 300, 30 / 300, fig.dpi_scale_trans)
with mpl.rc_context(
        {'font.family': 'Arial', 'font.size': 7, 'hatch.linewidth': .5, 'lines.linewidth': .3, 'patch.linewidth': .3,
         'xtick.labelsize': 6, 'ytick.labelsize': 6}):
    # A panel
    ax = fig.add_subplot(spec[0, 0:6])
    ax.text(-0.06, 0.9, 'A', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')

    x = np.arange(len(df_seq_count))
    print(x)
    metrics = {
        'BV': df_seq_count['BV_count'],
        'BY': df_seq_count['BY_count'],
    }

    # colors = ['#a9373b', '#2369bd']
    colors = ['#62B197', '#E18E6D']

    width = 0.4  # the width of the bars
    multiplier = 0

    i = 0
    for attribute, measurement in metrics.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[i], zorder=5)
        multiplier += 1
        i += 1

    ticklabels = df_BV_seq['year_month'].to_list()
    ax.set_xlim(-width, df_seq_count.shape[0] + 2)
    ax.tick_params(axis='both', which='major')
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=12))
    ax.set_xticks(np.arange(3 + width * 0.5, df_seq_count.shape[0], 12), (ticklabels)[3::12], rotation=0)
    ax.set_ylim(0, 2500)
    ax.set_yticks(np.arange(0, 2501, 500))
    ax.set_ylabel('No. of sequences')
    # ax.grid(axis='y', ls='--', zorder=0, alpha=0.5)
    ax.legend(['B/Victoria', 'B/Yamagata'], frameon=False, ncols=2, fontsize=6,columnspacing=0.8,
              loc='lower left', bbox_to_anchor=(0, 0.85), handlelength=2, handleheight=0.1)

    # ax.axvspan(51.8, 63.2, facecolor='#ef8c3b', edgecolor='none', alpha=.2)
    ax.axvspan(51.8, 69.2, facecolor='#ef8c3b', edgecolor='none', alpha=.2)
    [ax.spines[loc].set_visible(False) for loc in ax.spines if loc != 'bottom']
    axR = ax.twinx()
    axR.plot(df_case['time'], df_case['BVIC_NODEL'], label='B/Victoria cases', color='#62B197', linewidth=1, zorder=100)
    axR.plot(df_case['time'], df_case['BYAM'], label='B/Yamagata cases', color='#E18E6D', linewidth=1, zorder=100)
    # axR.set_xlim(pd.Timestamp('2017-10-01'), pd.Timestamp('2024-03-31'))  # 设置 x 轴日期范围
    axR.set_ylim(0, 8000)  # 设置 x 轴日期范围
    axR.set_yticks([0, 2000, 4000, 6000, 8000])
    axR.set_ylabel('No. of positive specimens')
    # legend of NPIs
    legend_elements = [
        Line2D([0], [0], color='#ef8c3b', alpha=.2, lw=2, linestyle='-', label='Acute phase of the COVID-19 pandemic')
    ]
    axR.legend(handles=legend_elements, frameon=False, ncols=1, fontsize=6, loc='lower left',
              bbox_to_anchor=(0.226, 0.85),
              columnspacing=0.8, handletextpad=0.5, handlelength=1.5)
    [axR.spines[loc].set_visible(False) for loc in axR.spines if loc != 'bottom']

    # B panel
    ax = fig.add_subplot(spec[2, 0])
    ax.text(-0.06, 0.6, 'D', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')

    # Plot world map
    df = pd.read_csv(
        r"./data/BV_country_clade_early.csv", index_col=0)
    # Dictionary of countries and corresponding antigenic clade counts
    country_dct = dict(zip(df['country'], df[list(df.columns)[1:-1]].apply(list, axis=1)))

    # Colors
    variant_lst = ['V1A', 'V1A.1', 'V1A.3', 'V1A.3a', 'V1A.3a.1', 'V1A.3a.2']
    color_dct = {'V1A': '#4E6FFF', 'V1A.1': '#5FB3F3', 'V1A.2': '#7FDBBD', 'V1A.3': '#B1EC84', 'V1A.3a': '#E8ED5D',
                 'V1A.3a.1': '#FFD84C', 'V1A.3a.2': '#FF993F'}
    legend_order = ['V1A', 'V1A.1', 'V1A.3', 'V1A.3a', 'V1A.3a.1', 'V1A.3a.2']

    # Plot world map
    ax = Geoplot(ax)
    ax.set_title("Pre-COVID-19", fontsize=6)

    # C panel
    ax = fig.add_subplot(spec[2, 1])
    ax.text(-0.06, 0.6, 'E', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')


    df = pd.read_csv(
        r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BV_gasaid/BV_region_seq/data_process/"
        r"final/figure_data/fig1/geo/BV_country_clade_late.csv", index_col=0)

    country_dct = dict(zip(df['country'], df[list(df.columns)[1:-1]].apply(list, axis=1)))

    variant_lst = ['V1A', 'V1A.3', 'V1A.3a', 'V1A.3a.1', 'V1A.3a.2']
    color_dct = {'V1A': '#4E6FFF', 'V1A.2': '#7FDBBD', 'V1A.3': '#B1EC84', 'V1A.3a': '#E8ED5D',
                 'V1A.3a.1': '#FFD84C', 'V1A.3a.2': '#FF993F'}
    legend_order = ['V1A', 'V1A.3', 'V1A.3a', 'V1A.3a.1', 'V1A.3a.2']


    ax = Geoplot(ax)
    ax.set_title("COVID-19", fontsize=6)

    # D panel
    ax = fig.add_subplot(spec[2, 2])
    ax.text(-0.06, 0.6, 'F', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')


    df = pd.read_csv(
        r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BY_gasaid/BY_region_seq/data_process/"
        r"final/figure_data/fig1/geo/BY_country_clade_early.csv", index_col=0)

    country_dct = dict(zip(df['country'], df[list(df.columns)[1:-1]].apply(list, axis=1)))


    variant_lst = ['Y3']
    color_dct = {'Y3': '#FFE753'}
    legend_order = ['Y3']


    ax = Geoplot(ax)
    ax.set_title("Pre-COVID-19", fontsize=6)

    # E panel
    ax = fig.add_subplot(spec[2, 3])
    ax.text(-0.06, 0.6, 'G', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')


    df = pd.read_csv(
        r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BY_gasaid/BY_region_seq/data_process/"
        r"final/figure_data/fig1/geo/BY_country_clade_late.csv", index_col=0)

    country_dct = dict(zip(df['country'], df[list(df.columns)[1:-1]].apply(list, axis=1)))


    variant_lst = ['Y3']
    color_dct = {'Y3': '#FFE753'}
    legend_order = ['Y3']


    ax = Geoplot(ax)
    ax.set_title("COVID-19", fontsize=6)

    # A panel
    ax = fig.add_subplot(spec[1, 0:2])
    ax.text(-0.13, 0.93, 'B', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    df_clade_BV = pd.read_csv(
        r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BV_gasaid/BV_region_seq/data_process/"
        r"final/figure_data/fig1/time_clade/BV_time_clade_fig1.csv", index_col=0)
    df_clade_BV['time_plot'] = df_clade_BV['year_month'].apply(lambda x: date_num(x))
    print(df_clade_BV)

    df_clade_BV['year_month'] = pd.to_datetime(df_clade_BV['year_month'])
    df_clade_BV = df_clade_BV[df_clade_BV['year_month'] >= '2013-01-01'].copy()
    df_clade_BV.reset_index(inplace=True, drop=True)

    x = np.array(df_clade_BV['time_plot'])
    y1 = np.array(df_clade_BV['V1A'])
    y2 = np.array(df_clade_BV['V1A.1'])
    y3 = np.array(df_clade_BV['V1A.2'])
    y4 = np.array(df_clade_BV['V1A.3'])
    y5 = np.array(df_clade_BV['V1A.3a'])
    y6 = np.array(df_clade_BV['V1A.3a.1'])
    y7 = np.array(df_clade_BV['V1A.3a.2'])
    y8 = np.array(df_clade_BV['V1B'])

    cluster_colors = ['#4E6FFF', '#5FB3F3', '#7FDBBD', '#B1EC84', '#E8ED5D', '#FFD84C', '#FF993F', '#FF3B2D']
    cluster_labels = ['V1A', 'V1A.1', 'V1A.2', 'V1A.3', 'V1A.3a', 'V1A.3a.1', 'V1A.3a.2', 'V1B']

    f_y1 = interp1d(x, y1, kind='cubic')
    f_y2 = interp1d(x, y2, kind='cubic')
    f_y3 = interp1d(x, y3, kind='cubic')
    f_y4 = interp1d(x, y4, kind='cubic')
    f_y5 = interp1d(x, y5, kind='cubic')
    f_y6 = interp1d(x, y6, kind='cubic')
    f_y7 = interp1d(x, y7, kind='cubic')
    f_y8 = interp1d(x, y8, kind='cubic')


    x_interp = np.linspace(x.min(), x.max(), 10000)
    y1_interp = f_y1(x_interp)
    y2_interp = f_y2(x_interp)
    y3_interp = f_y3(x_interp)
    y4_interp = f_y4(x_interp)
    y5_interp = f_y5(x_interp)
    y6_interp = f_y6(x_interp)
    y7_interp = f_y7(x_interp)
    y8_interp = f_y8(x_interp)

    cluster_colors.reverse()
    cluster_labels.reverse()
    ax.stackplot(x_interp, y8_interp, y7_interp, y6_interp, y5_interp, y4_interp, y3_interp, y2_interp, y1_interp,
             labels=cluster_labels, colors=cluster_colors, alpha=0.9, baseline='sym', edgecolor='k')

    ax.set_xlim(2012.8, 2024.2)
    ax.set_xticks(np.arange(2013, 2025, 2).tolist())
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.set_ylim(-1200, 1200)
    ax.set_yticks(np.arange(-1200, 910, 300).tolist())
    ax.set_title("B/Victoria", fontsize=6)
    # ax.set_yticklabels([])
    ax.axvspan(2020+(1/12), 2021+(7/12), ymax=0.93, facecolor='#ef8c3b', edgecolor='none', alpha=.2)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), ncol=4, frameon=False, loc='lower left', bbox_to_anchor=(0, 0.6),fontsize=6,
              columnspacing=.6, handletextpad=0.25,handlelength=0.7)
    [ax.spines[loc].set_visible(False) for loc in ax.spines if loc != 'bottom']

    # D panel
    ax = fig.add_subplot(spec[1, 2:4])
    ax.text(-0.13, 0.93, 'C', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    df_clade_BY = pd.read_csv(
        r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BY_gasaid/BY_region_seq/data_process/"
        r"final/figure_data/fig1/time_clade/BY_time_clade_fig1.csv", index_col=0)
    df_clade_BY['time_plot'] = df_clade_BY['year_month'].apply(lambda x: date_num(x))
    print(df_clade_BY)

    df_clade_BY['year_month'] = pd.to_datetime(df_clade_BY['year_month'])
    df_clade_BY = df_clade_BY[df_clade_BY['year_month'] >= '2013-01-01'].copy()
    df_clade_BY.reset_index(inplace=True, drop=True)

    x = np.array(df_clade_BY['time_plot'])
    y1 = np.array(df_clade_BY['Y1'])
    y2 = np.array(df_clade_BY['Y2'])
    y3 = np.array(df_clade_BY['Y3'])
    cluster_colors = ['#5CABF8', '#9FE895', '#FFE753']
    cluster_labels = ['Y1', 'Y2', 'Y3']

    f_y1 = interp1d(x, y1, kind='cubic')
    f_y2 = interp1d(x, y2, kind='cubic')
    f_y3 = interp1d(x, y3, kind='cubic')

    x_interp = np.linspace(x.min(), x.max(), 10000)
    y1_interp = f_y1(x_interp)
    y2_interp = f_y2(x_interp)
    y3_interp = f_y3(x_interp)

    cluster_colors.reverse()
    cluster_labels.reverse()
    print(cluster_colors)
    ax.stackplot(x_interp, y3_interp, y2_interp, y1_interp,
             labels=cluster_labels, colors=cluster_colors, alpha=0.9, baseline='sym', edgecolor='k')

    ax.set_xlim(2012.8, 2024.2)
    ax.set_xticks(np.arange(2013, 2025, 2).tolist())
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.set_ylim(-1100, 1100)
    ax.set_yticks(np.arange(-1100, 1100, 300).tolist())
    ax.set_title("B/Yamagata", fontsize=6)
    ax.axvspan(2020+(1/12), 2021+(7/12), ymax=0.95, facecolor='#ef8c3b', edgecolor='none', alpha=.2)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), ncol=3, frameon=False, loc='lower left', bbox_to_anchor=(0, 0.75), fontsize=6,
              columnspacing=.6, handletextpad=0.25, handlelength=0.7)
    [ax.spines[loc].set_visible(False) for loc in ax.spines if loc != 'bottom']

    # h panel
    ax = fig.add_subplot(spec[3, 0:2])
    ax.text(-0.13, 0.93, 'H', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    ax.bar(range(2013,2025), [1], width=1, alpha=0.8,
           color=['#ED7A75']*6 + ['#74AEDA']*2 + ['#7EB244']*2 + ['#9A7ABB']*2)
    # color = ['#ED7A75', '#ED7A75', '#ED7A75', '#ED7A75', '#ED7A75', '#ED7A75', '#28C6B8', '#28C6B8', '#7EB244',
    #          '#7EB244', '#9A7ABB', '#9A7ABB']
    ax.set_xlim(2012.5, 2024.5)
    ax.set_xticks(np.arange(2013, 2025,2).tolist())
    ax.set_xlim(2012.8, 2024.2)
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.text(2015, 0.2, 'BR08', c='w')
    ax.text(2019, 0.2, 'CO17', c='w')
    ax.text(2021, 0.2, 'WA19', c='w')
    ax.text(2022.85, 0.2, 'AU21', c='w')
    ax.tick_params(axis='y', which='both', length=0)
    [ax.spines[loc].set_visible(False) for loc in ax.spines if loc != 'bottom']

    # i panel
    ax = fig.add_subplot(spec[3, 2:4])
    ax.text(-0.13, 0.93, 'I', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    # ax.bar(range(2013, 2025), [1], width=1, alpha=0.85,
    #        color=['#5CABF8', '#9FE895', '#9FE895', '#FFE753', '#FFE753', '#FFE753', '#FFE753', '#FFE753', '#FFE753',
    #               '#FFE753', '#FFE753', '#FFE753'])
    ax.bar(range(2013, 2025), [1], width=1, alpha=0.8,
           color=['#A2D2BF']*1 + ['#E9C46A']*2 + ['#D87659']*9)
    # color = ['#6CD1C3', '#C6A1CC', '#C6A1CC', '#E89164', '#E89164', '#E89164', '#E89164', '#E89164', '#E89164',
    #          '#E89164', '#E89164', '#E89164']
    ax.set_xlim(2012.8, 2024.2)
    ax.set_xticks(np.arange(2013, 2025, 2).tolist())
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.text(2012.8, 0.23, 'WI10', c='w', fontsize=5)
    ax.text(2014, 0.2, 'MA12', c='w')
    ax.text(2019, 0.2, 'PH13', c='w')
    ax.tick_params(axis='y', which='both', length=0)
    [ax.spines[loc].set_visible(False) for loc in ax.spines if loc != 'bottom']
    fig.get_layout_engine().set(w_pad=10 / 300, h_pad=1/ 300, wspace=0.01, hspace=0.1)

    fig.get_layout_engine().set(w_pad=2/300, h_pad=2/300, hspace=0.02, wspace=0.02)
plt.savefig(r"./figure/fig1.pdf")
plt.show()


