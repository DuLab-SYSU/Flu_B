#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime as dt
from datetime import timedelta
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.transforms import ScaledTranslation
from matplotlib.ticker import AutoMinorLocator, AutoLocator
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from matplotlib.transforms import ScaledTranslation
mpl.rcParams['pdf.fonttype'] = 42

# transfer week to num time
def step_time(step):
    
    '''transfer step to time'''

    step_time = 2015 + (43 / 52) + ((step) / 52)
    
    return step_time

# surveillance data
data = pd.read_csv(r"../../../../data/flu_incidence_us.csv", index_col = 0)
data['Victoria_IR'] = data['ILI%'] * data['positive_rate'] * data['Victoria_proporation'] * 100000
data['Yamagata_IR'] = data['ILI%'] * data['positive_rate'] * data['Yamagata_proporation'] * 100000
data['Victoria_IR_rolling'] = data['Victoria_IR'].rolling(window=4).mean().round()
data['Yamagata_IR_rolling'] = data['Yamagata_IR'].rolling(window=4).mean().round()
data = data.dropna(subset=['Victoria_IR_rolling','Yamagata_IR_rolling'])
# data = data.iloc[:, :].copy()
data.reset_index(inplace=True, drop=True)
data[['Victoria_IR_rolling','Yamagata_IR_rolling']] = data[['Victoria_IR_rolling','Yamagata_IR_rolling']].replace(0, 1e-99)
data['step'] = np.arange(len(data))
data['time_plot'] = data.apply(lambda row: step_time(row['step']), axis=1)
data

# simmulation data under different BV escape
df_v0 = pd.read_csv(r"./data/scenario/data_simmulation_orign_v0.00.csv")

# 95% simmulation data
df_ci = pd.read_csv("./result/fluB_ci/cases_ci.csv", index_col=0)

# model fitting figure
cm = 2.54
N = 100000.
# plot
fig = plt.figure(figsize=(5.7/ cm, 6/ cm), dpi=300, layout="constrained")
spec = fig.add_gridspec(2, 1)
trans = ScaledTranslation(-30 / 300, 30 / 300, fig.dpi_scale_trans)
with mpl.rc_context(
        {'font.family': 'Arial', 'font.size': 7, 'hatch.linewidth': .5, 'lines.linewidth': .3, 'patch.linewidth': .3,
         'xtick.labelsize': 6, 'ytick.labelsize': 6}):
    
    # A panel-- Fit of BV
    ax = fig.add_subplot(spec[0, 0])
    ax.text(-0.15, 0.93, 'a', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    
    ax.plot(data.loc[:153, 'time_plot'], data.loc[:153, "Victoria_IR_rolling"], linewidth=1, color='#2C357F', 
            label='Data', zorder=100, alpha=1.)
    ax.plot(df_v0.loc[1:154, 'time_plot'], df_v0.loc[1:154, "BV_y0.00_n0.00"], linewidth=1, color='#2C357F', 
            label='Fit', ls='--', zorder=100, alpha=1.)
    ax.fill_between(df_v0.loc[1:154, 'time_plot'], df_ci.loc[:153, 'bv_lower'], df_ci.loc[:153, 'bv_upper'], color='#2C357F', alpha=0.3)
    ax.set_xlim(2015.7, 2018+45/52)
    ax.set_xticks(np.arange(2016, 2019, 1))
    ax.set_ylim(0, 120)
    ax.set_yticks(np.arange(0, 120, 20))
    ax.set_ylabel('Cases')
    ax.text(2017, 104, "Correlation=0.954", fontsize=5, color="#2C357F")
    ax.set_title("B/Victoria", fontsize=7)
    ax.grid(axis='y', ls='--', zorder=0, alpha=0.3)
    
    ax.legend(frameon=False, ncols=2, fontsize=5,loc='lower left', bbox_to_anchor=(0, 0.75), columnspacing=0.8, handletextpad=0.5, handlelength=1.5,)

    # B panel-- Fit of BY
    ax = fig.add_subplot(spec[1, 0])
    ax.text(-0.15, 0.91, 'b', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    
    ax.plot(data.loc[:153, 'time_plot'], data.loc[:153, "Yamagata_IR_rolling"], linewidth=1, color='#AB0C1D', 
            label='Data', zorder=100, alpha=1.)
    ax.plot(df_v0.loc[1:154, 'time_plot'], df_v0.loc[1:154, "BY_y0.00_n0.00"], linewidth=1, color='#AB0C1D', 
            label='Fit', ls='--', zorder=100, alpha=1.)
    ax.fill_between(df_v0.loc[1:154, 'time_plot'], df_ci.loc[:153, 'by_lower'], df_ci.loc[:153, 'by_upper'], color='#AB0C1D', alpha=0.3)
    ax.set_xlim(2015.7, 2018+45/52)
    ax.set_xticks(np.arange(2016, 2019, 1))
    ax.set_ylim(0, 1000)
    ax.set_yticks(np.arange(0, 1000, 200))
    ax.set_ylabel('Cases')
    ax.set_xlabel('Date')
    ax.set_title("B/Yamagata", fontsize=7)
    ax.grid(axis='y', ls='--', zorder=0, alpha=0.3)
    ax.text(2017, 860, "Correlation=0.864", fontsize=5, color="#AB0C1D")
    ax.legend(frameon=False, ncols=2, fontsize=5,loc='lower left', bbox_to_anchor=(0, 0.75), columnspacing=0.8, handletextpad=0.5, handlelength=1.5,)

fig.get_layout_engine().set(w_pad=10/300, h_pad=10/300, hspace=0.02, wspace=0.02)
plt.savefig(r"./figure/fig_s15.pdf")