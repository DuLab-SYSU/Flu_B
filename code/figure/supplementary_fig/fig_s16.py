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

# import epidemic size data
df_size_nc = pd.read_csv(r"./data/scenario/by_extinction_size_changeC.csv", index_col=0) # ne represents NPIs and Cross protection
df_size_nc

df_size_nw = pd.read_csv(r"./data/scenario/by_extinction_size_changeW.csv", index_col=0) # ne represents NPIs and Cross protection
df_size_nw

df_size_nt = pd.read_csv(r"./data/scenario/by_extinction_size_changeT.csv", index_col=0) # ne represents NPIs and Cross protection
df_size_nt


# extract cv and cy of each combination
def extract_param_nc(com):
    cv = 1. - float(f"{float(com[7:11]):.2f}")
    cy = 1. - float(f"{float(com[-4:]):.2f}")
    return cv, cy

# extract wv and wy of each combination
def extract_param_nw(com):
    wv = float(f"{float(com[7:11]):.2f}")
    wy = float(f"{float(com[-4:]):.2f}")
    return wv, wy

# extract tv and ty of each combination
def extract_param_nt(com):
    try:
        tv = float(f"{float(com[7:11]):.1f}")
        ty = float(f"{float(com[-5:-1]):.1f}")
    except ValueError:
        tv = 0.0
        ty = 0.0
    return tv, ty

# extract npi and esp of each combination
df_size_ne['bv_esp'] = df_size_ne.apply(lambda row: extract_param_ne(row['combination'])[0], axis=1)
df_size_ne['by_esp'] = df_size_ne.apply(lambda row: extract_param_ne(row['combination'])[1], axis=1)
df_size_ne['npi'] = df_size_ne.apply(lambda row: extract_param_ne(row['combination'])[2], axis=1)
# df_size_ne = df_size_ne.query('npi <=0.5').copy()
# extract npi and R0 of each combination
df_size_nr['npi'] = df_size_nr.apply(lambda row: extract_param_nr(row['combination'])[0], axis=1)
df_size_nr['r0'] = df_size_nr.apply(lambda row: extract_param_nr(row['combination'])[1], axis=1)
# extract cv and cy of each combination
df_size_nc['cv'] = df_size_nc.apply(lambda row: extract_param_nc(row['combination'])[0], axis=1)
df_size_nc['cy'] = df_size_nc.apply(lambda row: extract_param_nc(row['combination'])[1], axis=1)
# extract wv and wy of each combination
df_size_nw['wv'] = df_size_nw.apply(lambda row: extract_param_nw(row['combination'])[0], axis=1)
df_size_nw['wy'] = df_size_nw.apply(lambda row: extract_param_nw(row['combination'])[1], axis=1)
# extract tv and ty of each combination
df_size_nt['tv'] = df_size_nt.apply(lambda row: extract_param_nt(row['combination'])[0], axis=1)
df_size_nt['ty'] = df_size_nt.apply(lambda row: extract_param_nt(row['combination'])[1], axis=1)

# compare to previous season size
size_base = df_v0["BY_y0.00_n0.00"].iloc[:206].sum()/4
df_size_ne['size_fold'] = df_size_ne['extinction_size'] / size_base
df_size_nr['size_fold'] = df_size_nr['extinction_size'] / size_base
df_size_nc['size_fold'] = df_size_nc['extinction_size'] / size_base
df_size_nw['size_fold'] = df_size_nw['extinction_size'] / size_base
df_size_nt['size_fold'] = df_size_nt['extinction_size'] / size_base

# determine the range of heatmap
print(df_size_ne['size_fold'].min(), df_size_ne['size_fold'].max())
print(df_size_nr['size_fold'].min(), df_size_nr['size_fold'].max())
print(df_size_nc['size_fold'].min(), df_size_nc['size_fold'].max())
print(df_size_nw['size_fold'].min(), df_size_nw['size_fold'].max())
print(df_size_nt['size_fold'].min(), df_size_nt['size_fold'].max())

# transfer the raw data if epidemic size to heatmap format data
data_by_size_nc = (
    df_size_nc
    .pivot(index="cv", columns="cy", values="size_fold")
)

df_size_nw = df_size_nw.query('wv<=0.7 & wy<=0.7').copy()

# transfer the raw data if epidemic size to heatmap format data
data_by_size_nw = (
    df_size_nw
    .pivot(index="wv", columns="wy", values="size_fold")
)

df_size_nt = df_size_nt.query('tv>=10. & tv<=29. & ty>=10. & ty<=29.').copy()

df_size_nt['tv'] = df_size_nt['tv'] - 9.
df_size_nt['ty'] = df_size_nt['ty'] - 9.

# transfer the raw data if epidemic size to heatmap format data
data_by_size_nt = (
    df_size_nt
    .pivot(index="tv", columns="ty", values="size_fold")
)

# heatmap-- cross protection of BV and BY
cm = 2.54
N = 100000.
# 画图
fig = plt.figure(figsize=(18.4/ cm, 6/ cm), dpi=300, layout="constrained")
spec = fig.add_gridspec(1, 3)
trans = ScaledTranslation(-30 / 300, 30 / 300, fig.dpi_scale_trans)
with mpl.rc_context(
        {'font.family': 'Arial', 'font.size': 7, 'hatch.linewidth': .5, 'lines.linewidth': .3, 'patch.linewidth': .3,
         'xtick.labelsize': 6, 'ytick.labelsize': 6}):

    # A panel
    ax = fig.add_subplot(spec[0, 1])
    ax.text(-0.15, 0.97, 'b', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=0.25, vmax=0.5)
    heatmap = sns.heatmap(data=data_by_size_nw, annot=False,
                norm = norm,
                cbar_kws={'shrink': 0.7, 'aspect': 30, 'ticks':[0, 0.2, 0.4, 0.5]},
                cmap="Blues", square=True, ax=ax)
    # ax.set_xlim(1, 52)
    ax.set_title('Amplitude', fontsize=8)
    n_rows, n_cols = data_by_size_nw.shape
    ax.set_xticks(np.arange(0., n_cols, 10)+0.5, np.round(np.arange(0., 0.8, 0.1), 1))
    ax.set_yticks(np.arange(0., n_rows, 10)+0.5, np.round(np.arange(0., 0.8, 0.1), 1))
    ax.invert_yaxis()
    ax.set_xlabel(r'$\omega_{BY}$')
    ax.set_ylabel(r'$\omega_{BV}$')
    ax.tick_params(axis='y', rotation=0)
    ax.tick_params(axis='x', rotation=0)
    # colorbar label 
    cbar = heatmap.collections[0].colorbar
    # cbar.ax.set_ylabel("Fold change of 2019/2020 epidemic size", rotation=90)

    # B panel
    ax = fig.add_subplot(spec[0, 2])
    ax.text(-0.15, 0.97, 'c', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=0.25, vmax=0.5)
    heatmap = sns.heatmap(data=data_by_size_nt, annot=False,
                norm = norm,
                cbar_kws={'shrink': 0.7, 'aspect': 30, 'ticks':[0, 0.2, 0.4, 0.5]},
                cmap="Blues", square=True, ax=ax)
    # ax.set_xlim(1, 52)
    ax.set_title('Peak week', fontsize=8)
    n_rows, n_cols = data_by_size_nt.shape
    ax.set_xticks([0.5, 40.5, 90.5, 140.5, 190.5], [1, 5, 10, 15, 20])
    ax.set_yticks([0.5, 40.5, 90.5, 140.5, 190.5], [1, 5, 10, 15, 20])
    # ax.set_xticks(np.arange(0., n_cols, 5)+0.5, np.round(np.arange(0., 21, 5), 1))
    # ax.set_yticks(np.arange(0., n_rows, 5)+0.5, np.round(np.arange(0., 21, 5), 1))
    ax.invert_yaxis()
    ax.set_xlabel(r'$t_0^{BY}$')
    ax.set_ylabel(r'$t_0^{BV}$')
    ax.tick_params(axis='y', rotation=0)
    ax.tick_params(axis='x', rotation=0)
    # colorbar label 
    cbar = heatmap.collections[0].colorbar
    cbar.ax.set_ylabel("Fold change of 2019/2020 epidemic size", rotation=90, fontsize=6)

    # C panel
    ax = fig.add_subplot(spec[0, 0])
    ax.text(-0.15, 0.97, 'a', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=0.25, vmax=0.5)
    heatmap = sns.heatmap(data=data_by_size_nc, annot=False,
                norm = norm,
                cbar_kws={'shrink': 0.7, 'aspect': 30, 'ticks':[0, 0.2, 0.4, 0.5]},
                cmap="Blues", square=True, ax=ax)
    # ax.set_xlim(1, 52)
    ax.set_title('Cross-protection', fontsize=8)
    n_rows, n_cols = data_by_size_nc.shape
    ax.set_xticks(np.arange(0., n_cols, 20)+0.5, np.round(np.arange(0., 1.1, 0.2), 1))
    ax.set_yticks(np.arange(0., n_rows, 20)+0.5, np.round(np.arange(0, 1.1, 0.2), 1))
    ax.invert_yaxis()
    ax.set_xlabel('$C_{BV,BY}$')
    ax.set_ylabel('$C_{BY,BV}$')
    ax.tick_params(axis='y', rotation=0)
    ax.tick_params(axis='x', rotation=0)
    # colorbar label 
    cbar = heatmap.collections[0].colorbar
    # cbar.ax.set_ylabel("Fold change of 2019/2020 epidemic size", rotation=90, fontsize=6)

plt.savefig(r"./figure/fig_s16.png")