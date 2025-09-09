# -*- coding: utf-8 -*-
# @Time    : 2025/3/22 6:18 PM
# @Author  : Hanwenjie
# @File    : fig_s6.py
# @IDE     : PyCharm
# @REMARKS : description text
import pandas as pd
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
mpl.rcParams['pdf.fonttype']=42
import math
from matplotlib.patches import Wedge


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

def Geoplot(ax, legend_u, legend_l):
    ax.axis('off')
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    world = world[(world.name != "Antarctica") & (world.name != "Fr. S. Antarctic Lands")]
    myplot = world.plot(
        ax=ax,
        linewidth=0.3,
        color='lightgrey', edgecolor='white',
        missing_kwds={"color": "lightgrey", "edgecolor": "white", "label": "Missing values"})

    rat = 1.8

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

    labelsize = 6
    s_alpha = 0.9
    s_width = 0.5
    label1 = ['>1000', '500-1000', '100-500', '<100']
    p_variant_lst = []
    for variant in legend_order:
        pi = ax.scatter([], [], c=color_dct[variant], s=20, edgecolor='k', linewidth=s_width, alpha=s_alpha, zorder=-1,
                        label=variant)
        p_variant_lst.append(pi)

    legend_ax2 = ax.legend(p_variant_lst, [cluster_name[cluster] for cluster in legend_order], frameon=False,
                           title='Clusters', title_fontsize=labelsize,
                           alignment='left',
                           labelspacing=0.2, loc='lower left', fontsize=labelsize,
                           bbox_to_anchor=(0.03, legend_u), )
    legend_labels2 = legend_ax2.get_texts()
    [label.set_fontname('arial') for label in legend_labels2]
    legend_ax2.get_title().set_fontname('arial')

    p1 = ax.scatter([], [], c='white', s=math.sqrt(5000) / rat, edgecolor='k', linewidth=s_width, alpha=s_alpha,
                    zorder=-1)
    p2 = ax.scatter([], [], c='white', s=math.sqrt(1000) / rat, edgecolor='k', linewidth=s_width, alpha=s_alpha,
                    zorder=-1)
    p3 = ax.scatter([], [], c='white', s=math.sqrt(500) / rat, edgecolor='k', linewidth=s_width, alpha=s_alpha,
                    zorder=-1)
    p4 = ax.scatter([], [], c='white', s=math.sqrt(100) / rat, edgecolor='k', linewidth=s_width, alpha=s_alpha,
                    zorder=-1)

    legend_ax1 = ax.legend([p1, p2, p3, p4], label1, frameon=False, title='Numbers', title_fontsize=labelsize,
                           alignment='left',
                           labelspacing=0.8, loc='lower left', fontsize=labelsize,
                           bbox_to_anchor=(0.03, legend_l))
    legend_abelss = legend_ax1.get_texts()
    [label.set_fontname('arial') for label in legend_abelss]
    legend_ax1.get_title().set_fontname('arial')

    ax.add_artist(legend_ax2)

    myplot.set_axis_off()

    return ax

# specify unit conversion to cm
cm = 2.54

fig = plt.figure(figsize=(18/cm, 14.4/cm), dpi=300, layout="constrained")
spec = fig.add_gridspec(2,1)
trans = ScaledTranslation(-30/300, 30/300, fig.dpi_scale_trans)
with mpl.rc_context({'font.family': 'Arial', 'font.size': 7, 'hatch.linewidth': .5, 'lines.linewidth': .3, 'patch.linewidth': .3, 'xtick.labelsize': 6, 'ytick.labelsize': 6}):
    # A panel
    # Load world map data from the built-in geopandas dataset
    ax = fig.add_subplot(spec[0, 0])
    ax.text(0.0, 1.0, 'A', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom', fontfamily='Arial')
    df = pd.read_csv(
        r"./data/BV_map.csv", index_col=0)
    # Dictionary of countries and corresponding antigenic cluster counts
    country_dct = dict(zip(df['country'], df[['cluster' + str(i) + '_count' for i in range(1, 8)]].apply(list, axis=1)))

    # Colors
    variant_lst = ['cluster1', 'cluster2', 'cluster3', 'cluster4', 'cluster5', 'cluster6', 'cluster7']
    color_dct = {'cluster1': '#5c6fd5', 'cluster2': '#e67932', 'cluster3': '#8cbb69',
                 'cluster4': '#5d56cc', 'cluster5': '#65ae96', 'cluster6': '#dcab3c', 'cluster7': '#959797'}
    cluster_name = {'cluster1': 'BR08', 'cluster2': 'AU21', 'cluster3': 'WA19',
                    'cluster4': 'SD97', 'cluster5': 'CO17', 'cluster6': 'COV20', 'cluster7': 'BJ87'}
    legend_order = ['cluster7', 'cluster4', 'cluster1', 'cluster5', 'cluster3', 'cluster6', 'cluster2']

    # Draw world map
    ax = Geoplot(ax,0.35, 0.0)

    ax.set_title('B/Victoria')

    # B panel
    # Load world map data from the built-in geopandas dataset
    ax = fig.add_subplot(spec[1, 0])
    ax.text(0.0, 1.0, 'B', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom', fontfamily='Arial')
    df = pd.read_csv(
        r"/home/hanwenjie/Project/PAV-Bsubtype/data/original_data/BY_gasaid/BY_region_seq/data_process/final/"
        r"figure_data/geo_map/BY_map.csv", index_col=0)
    # Dictionary of countries and corresponding antigenic cluster counts
    country_dct = dict(zip(df['country'], df[['cluster' + str(i) + '_count' for i in range(1, 4)]].apply(list, axis=1)))

    # Colors
    variant_lst = ['cluster1', 'cluster2', 'cluster3']
    color_dct = {'cluster1': '#cbb742', 'cluster2': '#7eb876', 'cluster3': '#4988c5'}
    cluster_name = {'cluster1': 'WI10', 'cluster2': 'BJ93', 'cluster3': 'YA88'}
    legend_order = ['cluster3', 'cluster2', 'cluster1']

    # Draw world map
    ax = Geoplot(ax,0.5,0.15)

    ax.set_title('B/Yamagata')

plt.savefig(r"./figure/fig_s6.pdf")
plt.show()
