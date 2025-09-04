# -*- coding: utf-8 -*-
# @Time    : 2025/3/31 21:03
# @Author  : Hanwenjie
# @project : BY_dn_ds_cluster_MEME.py
# @File    : fig1_v10.py
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
    # Define the start date
    start_date = datetime(date.year, 1, 1)

    # Calculate the difference in days between dates
    days_diff = (date - start_date).days

    # Get the start year
    start_year = start_date.year

    # Calculate the number of days in the year
    days_in_year = 365 if (start_year % 4 == 0 and (start_year % 100 != 0 or start_year % 400 == 0)) else 366

    # Calculate the fraction of the year
    year_ratio = days_diff / days_in_year

    # Convert date to decimal year
    date_as_decimal = start_year + year_ratio

    return date_as_decimal


def drawPieMarker(ax, x, y, ratios, sizes, colors):
    assert sum(ratios) <= 1, 'sum of ratios needs to be <= 1'

    markers = []
    previous = 0
    for color, ratio in zip(colors, ratios):
        this = 360 * ratio + previous
        marker = Wedge(center=(x, y), r=sizes, theta1=previous, theta2=this, facecolor=color, edgecolor='k',linewidth=0.15)
        markers.append(marker)
        previous = this

    for marker in markers:
        ax.add_patch(marker)

    return ax
def Geoplot(ax):
    ax.axis('off')
    # Load world map
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
    # Convert date string to decimal month representation
    date_number = int(date[:4]) + (int(date[-2:]) - 1) / 12
    return date_number


# Import case data
df_case = pd.read_csv(
    r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BV_gasaid/BV_region_seq/data_process/final/"
    r"latest/figure_data/case/BV_BY_case.csv", index_col=0)
df_case['date'] = pd.to_datetime(df_case['date'])
df_case = df_case[df_case['date'] >= '2015-10-1'].copy()
df_case.reset_index(inplace=True, drop=True)
for i in range(df_case.shape[0]):
    df_case.loc[i, 'time'] = df_case.index[i] * (102 / 438)

print(df_case)

# Import sequence count data
df_BV_seq = pd.read_csv(
    r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BV_gasaid/BV_region_seq/data_process/final/"
    r"latest/figure_data/seq/BV_seq_count.csv", index_col=0)
df_BY_seq = pd.read_csv(
    r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BY_gasaid/BY_region_seq/data_process/final/"
    r"latest/figure_data/seq/BY_seq_count.csv", index_col=0)
df_BV_seq = df_BV_seq[df_BV_seq['year_month'] >= '2015-10'].copy()
df_BV_seq.reset_index(inplace=True, drop=True)
df_BY_seq = df_BY_seq[df_BY_seq['year_month'] >= '2015-10'].copy()
df_BY_seq.reset_index(inplace=True, drop=True)

# Merge BV and BY sequence counts
df_seq_count = pd.merge(df_BV_seq, df_BY_seq, on='year_month', how='outer')
df_seq_count.reset_index(inplace=True, drop=True)
df_seq_count = df_seq_count.fillna(0)
print(df_seq_count)
# df_seq_count.to_csv(r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BV_gasaid/BV_region_seq/data_process/final/"
#                     r"latest/figure_data/seq/BV_BY_seq_count.csv")


# Set centimeters conversion unit
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
    # Panel A
    ax = fig.add_subplot(spec[0, 0:6])
    ax.text(-0.06, 0.9, 'A', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')

    x = np.arange(len(df_seq_count))
    print(x)
    metrics = {
        'BV': df_seq_count['BV_count'],
        'BY': df_seq_count['BY_count'],
    }

    # Bar colors
    colors = ['#62B197', '#E18E6D']

    width = 0.4  # Bar width
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

    # Highlight COVID-19 period
    ax.axvspan(51.8, 69.2, facecolor='#ef8c3b', edgecolor='none', alpha=.2)
    [ax.spines[loc].set_visible(False) for loc in ax.spines if loc != 'bottom']
    axR = ax.twinx()
    axR.plot(df_case['time'], df_case['BVIC_NODEL'], label='B/Victoria cases', color='#62B197', linewidth=1, zorder=100)
    axR.plot(df_case['time'], df_case['BYAM'], label='B/Yamagata cases', color='#E18E6D', linewidth=1, zorder=100)
    axR.set_ylim(0, 8000)  # Set y-axis range
    axR.set_yticks([0, 2000, 4000, 6000, 8000])
    axR.set_ylabel('No. of positive specimens')
    # Legend for NPIs
    legend_elements = [
        Line2D([0], [0], color='#ef8c3b', alpha=.2, lw=2, linestyle='-', label='Acute phase of the COVID-19 pandemic')
    ]
    axR.legend(handles=legend_elements, frameon=False, ncols=1, fontsize=6, loc='lower left',
              bbox_to_anchor=(0.226, 0.85),
              columnspacing=0.8, handletextpad=0.5, handlelength=1.5)
    [axR.spines[loc].set_visible(False) for loc in axR.spines if loc != 'bottom']

# The rest of the code follows the same translation pattern for panels B-I
