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
df_v1 = pd.read_csv(r"./data/scenario/data_simmulation_orign_v0.01.csv")
df_v2 = pd.read_csv(r"./data/scenario/data_simmulation_orign_v0.02.csv")
df_v3 = pd.read_csv(r"./data/scenario/data_simmulation_orign_v0.03.csv")
df_v4 = pd.read_csv(r"./data/scenario/data_simmulation_orign_v0.04.csv")
df_v5 = pd.read_csv(r"./data/scenario/data_simmulation_orign_v0.05.csv")


# simmulation data under different R0 of 2018-2019 season
df_r05 = pd.read_csv(r"./data/scenario/data_simmulation_orign_v0.05_r1.05.csv")
df_r10 = pd.read_csv(r"./data/scenario/data_simmulation_orign_v0.05_r1.10.csv")

# import epidemic size data
df_size_ne = pd.read_csv(r"./data/scenario/by_extinction_size.csv", index_col=0) # ne represents NPIs and Escape
df_size_ne


# import epidemic size data
df_size_nr = pd.read_csv(r"./data/scenario/by_extinction_size_changeR0.csv", index_col=0) # ne represents NPIs and R0
df_size_nr


# extract npi and esp of each combination
def extract_param_ne(com):
    bv_esp = float(f"{float(com[1:5]):.2f}")
    by_esp = float(f"{float(com[10:15]):.3f}")
    npi = float(f"{float(com[-4:]):.2f}")
    return bv_esp, by_esp, npi

# extract npi and R0 of each combination
def extract_param_nr(com):
    npi = float(f"{float(com[10:14]):.2f}")
    r0 = float(f"{float(com[-5:]):.3f}")
    return npi, r0


# extract npi and esp of each combination
df_size_ne['bv_esp'] = df_size_ne.apply(lambda row: extract_param_ne(row['combination'])[0], axis=1)
df_size_ne['by_esp'] = df_size_ne.apply(lambda row: extract_param_ne(row['combination'])[1], axis=1)
df_size_ne['npi'] = df_size_ne.apply(lambda row: extract_param_ne(row['combination'])[2], axis=1)
# df_size_ne = df_size_ne.query('npi <=0.5').copy()
# extract npi and R0 of each combination
df_size_nr['npi'] = df_size_nr.apply(lambda row: extract_param_nr(row['combination'])[0], axis=1)
df_size_nr['r0'] = df_size_nr.apply(lambda row: extract_param_nr(row['combination'])[1], axis=1)

# compare to previous season size
size_base = df_v0["BY_y0.00_n0.00"].iloc[:206].sum()/4
df_size_ne['size_fold'] = df_size_ne['extinction_size'] / size_base
df_size_nr['size_fold'] = df_size_nr['extinction_size'] / size_base

# determine the range of heatmap
print(df_size_ne['size_fold'].min(), df_size_ne['size_fold'].max())
print(df_size_nr['size_fold'].min(), df_size_nr['size_fold'].max())

# transfer the raw data if epidemic size to heatmap format data
df_by_figsize = df_size_ne.query("bv_esp == 0.10").copy()
data_by_size_ne = (
    df_by_figsize
    .pivot(index="npi", columns="by_esp", values="size_fold")
)

# transfer the raw data if epidemic size to heatmap format data
data_by_size_nr = (
    df_size_nr
    .pivot(index="npi", columns="r0", values="size_fold")
)

# fig 5--main text
# specify unit conversion to cm
cm = 2.54
N = 100000.
# plotting
fig = plt.figure(figsize=(18.4/ cm, 15/ cm), dpi=300, layout="constrained")
spec = fig.add_gridspec(4, 2, width_ratios=[1.3, 1])
trans = ScaledTranslation(-30 / 300, 30 / 300, fig.dpi_scale_trans)
with mpl.rc_context(
        {'font.family': 'Arial', 'font.size': 7, 'hatch.linewidth': .5, 'lines.linewidth': .3, 'patch.linewidth': .3,
         'xtick.labelsize': 6, 'ytick.labelsize': 6}):

    # A panel-- no npis and no esp
    ax = fig.add_subplot(spec[0, 0])
    ax.text(-0.1, 0.95, 'a', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')

    ax.plot(df_v5.loc[:, 'time_plot'], df_v5.loc[:, "BV_y0.00_n0.10"]/N, linewidth=1, color='#2C357F', 
            alpha=1., label='BV (BV_esp=0.05)', zorder=100)
    ax.plot(df_v5.loc[:, 'time_plot'], df_v5.loc[:, "BY_y0.00_n0.10"]/N, linewidth=1, color='#AB0C1D', 
            alpha=1., label='BY (BV_esp=0.05)', zorder=100) 

    # ax.plot(df_v3.loc[206:, 'time_plot'], df_v3.loc[206:, "BV_y0.00_n0.10"]/0.3/N, linewidth=1, color='#2C357F', 
    #         alpha=0.6, label='BV (BV_esp=0.02)', zorder=100)
    # ax.plot(df_v3.loc[206:, 'time_plot'], df_v3.loc[206:, "BY_y0.00_n0.10"]/0.3/N, linewidth=1, color='#AB0C1D', 
    #         alpha=0.6, label='BY (BV_esp=0.02)', zorder=100)
    # ax.plot(df_v5.loc[206:, 'time_plot'], df_v5.loc[206:, "BV_y0.00_n0.10"]/0.3/N, linewidth=1, color='#2C357F', 
    #         alpha=0.3, label='BV (BV_esp=0.05)', zorder=100)
    # ax.plot(df_v5.loc[206:, 'time_plot'], df_v5.loc[206:, "BY_y0.00_n0.10"]/0.3/N, linewidth=1, color='#AB0C1D', 
    #         alpha=0.3, label='BY (BV_esp=0.05)', zorder=100) 

    ax2 = ax.twinx()
    # ax2.plot(df_v0.loc[:206, 'time_plot'], df_v0.loc[:206, ["S_y0.00_n0.00", "Ry_y0.00_n0.00"]].sum(axis=1)/N, linewidth=1, 
    #          alpha=1, color='#2C357F', label='BV', ls = '--', zorder=100)
    # ax2.plot(df_v0.loc[:206, 'time_plot'], df_v0.loc[:206, ["S_y0.00_n0.00", "Rv_y0.00_n0.00"]].sum(axis=1)/N, linewidth=1, 
    #          alpha=1, color='#AB0C1D', label='BY', ls = '--', zorder=100)
    # ax2.plot(df_v3.loc[206:, 'time_plot'], df_v3.loc[206:, ["S_y0.00_n0.10", "Ry_y0.00_n0.10"]].sum(axis=1)/N, linewidth=1, 
    #          alpha=0.6, color='#2C357F', label='BV', ls = '--', zorder=100)
    # ax2.plot(df_v3.loc[206:, 'time_plot'], df_v3.loc[206:, ["S_y0.00_n0.10", "Rv_y0.00_n0.10"]].sum(axis=1)/N, linewidth=1, 
    #          alpha=0.6, color='#AB0C1D', label='BY', ls = '--', zorder=100)
    ax2.plot(df_v5.loc[:, 'time_plot'], df_v5.loc[:, ["S_y0.00_n0.10", "Ry_y0.00_n0.10"]].sum(axis=1)/N, linewidth=1, 
             alpha=1., color='#2C357F', label='BV', ls = '--', zorder=100)
    ax2.plot(df_v5.loc[:, 'time_plot'], df_v5.loc[:, ["S_y0.00_n0.10", "Rv_y0.00_n0.10"]].sum(axis=1)/N, linewidth=1, 
             alpha=1., color='#AB0C1D', label='BY', ls = '--', zorder=100)

    ax.set_xlim(2015.7, 2020+39/52)
    ax.set_xticks(np.arange(2016, 2021, 1))
    ax.set_ylim(0, 0.0215)
    ax.set_yticks(np.arange(0, 0.021, 0.005))
    ax.set_yticklabels(["0.0%", "0.5%", "1.0%", "1.5%", "2.0%"])
    ax.tick_params(axis='y', pad=1)
    # ax.set_xlabel('Date')
    ax.set_ylabel('Infectious')
    ax2.set_ylabel('Susceptible')
    ax2.set_ylim(0, 0.82)
    ax2.set_yticks(np.arange(0.4, 0.71, 0.1))
    ax2.set_yticklabels(["40%", "50%", "60%", "70%"])
    ax2.tick_params(axis='y', pad=1)

    ax.set_title("Real-world scenario", fontsize=7)
    ax.grid(axis='y', ls='--', zorder=0, alpha=0.3)
    ax.legend(frameon=False, ncols=1, fontsize=5,loc='lower left', bbox_to_anchor=(0, 0.75), columnspacing=0.8, handletextpad=0.5, handlelength=1.5,)
    # legend of infectious and susceptible
    legend_elements = [
        Line2D([0], [0], color='black', lw=1, linestyle='-', label='Infectious'),
        Line2D([0], [0], color='black', lw=1, linestyle='--', label='Susceptible')
    ]
    ax2.legend(handles=legend_elements, frameon=False, ncols=1, fontsize=5,loc='lower left', bbox_to_anchor=(0.3, 0.75), 
               columnspacing=0.8, handletextpad=0.5, handlelength=1.5)

    ax.axvspan(2019+40/52, 2021 + 8 / 12, facecolor='#FFEAC1', edgecolor='none', alpha=.6)
    ax.axvline(x=2021+11/12, zorder=0, color='#a9373b', ls='--', linewidth=1)
    # ax.text(x=2020.55, y=2500000, s='NPIs', c='#a9373b', fontsize=7)

    # B panel-- npis
    ax = fig.add_subplot(spec[1, 0])
    ax.text(-0.1, 0.95, 'b', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')

    ax.plot(df_v5.loc[:206, 'time_plot'], df_v5.loc[:206, "BV_y0.00_n0.00"]/N, linewidth=1, color='#2C357F', label='', 
            zorder=100, alpha=1.)
    ax.plot(df_v5.loc[:206, 'time_plot'], df_v5.loc[:206, "BY_y0.00_n0.00"]/N, linewidth=1, color='#AB0C1D', label='', 
            zorder=100, alpha=1.)

    ax.plot(df_v5.loc[206:, 'time_plot'], df_v5.loc[206:, "BV_y0.00_n0.05"]/N, linewidth=1, color='#2C357F', 
            alpha=0.6, label='BV (NPIs=0.05)', zorder=100)
    ax.plot(df_v5.loc[206:, 'time_plot'], df_v5.loc[206:, "BY_y0.00_n0.05"]/N, linewidth=1, color='#AB0C1D', 
            alpha=0.6, label='BY (NPIs=0.05)', zorder=100)
    ax.plot(df_v5.loc[206:, 'time_plot'], df_v5.loc[206:, "BV_y0.00_n0.00"]/N, linewidth=1, color='#2C357F', 
            alpha=0.3, label='BV (NPIs=0.00)', zorder=100)
    ax.plot(df_v5.loc[206:, 'time_plot'], df_v5.loc[206:, "BY_y0.00_n0.00"]/N, linewidth=1, color='#AB0C1D', 
            alpha=0.3, label='BY (NPIs=0.00)', zorder=100) 

    ax2 = ax.twinx()
    ax2.plot(df_v0.loc[:206, 'time_plot'], df_v0.loc[:206, ["S_y0.00_n0.00", "Ry_y0.00_n0.00"]].sum(axis=1)/N, linewidth=1, 
             alpha=1, color='#2C357F', label='BV', ls = '--', zorder=100)
    ax2.plot(df_v0.loc[:206, 'time_plot'], df_v0.loc[:206, ["S_y0.00_n0.00", "Rv_y0.00_n0.00"]].sum(axis=1)/N, linewidth=1, 
             alpha=1, color='#AB0C1D', label='BY', ls = '--', zorder=100)
    ax2.plot(df_v5.loc[206:, 'time_plot'], df_v5.loc[206:, ["S_y0.00_n0.05", "Ry_y0.00_n0.05"]].sum(axis=1)/N, linewidth=1, 
             alpha=0.6, color='#2C357F', label='BV', ls = '--', zorder=100)
    ax2.plot(df_v5.loc[206:, 'time_plot'], df_v5.loc[206:, ["S_y0.00_n0.05", "Rv_y0.00_n0.05"]].sum(axis=1)/N, linewidth=1, 
             alpha=0.6, color='#AB0C1D', label='BY', ls = '--', zorder=100)
    ax2.plot(df_v5.loc[206:, 'time_plot'], df_v5.loc[206:, ["S_y0.00_n0.00", "Ry_y0.00_n0.00"]].sum(axis=1)/N, linewidth=1, 
             alpha=0.3, color='#2C357F', label='BV', ls = '--', zorder=100)
    ax2.plot(df_v5.loc[206:, 'time_plot'], df_v5.loc[206:, ["S_y0.00_n0.00", "Rv_y0.00_n0.00"]].sum(axis=1)/N, linewidth=1, 
             alpha=0.3, color='#AB0C1D', label='BY', ls = '--', zorder=100)

    ax.set_xlim(2015.7, 2020+39/52)
    ax.set_xticks(np.arange(2016, 2021, 1))
    ax.set_ylim(0, 0.0215)
    ax.set_yticks(np.arange(0, 0.021, 0.005))
    ax.set_yticklabels(["0.0%", "0.5%", "1.0%", "1.5%", "2.0%"])
    ax.tick_params(axis='y', pad=1)
    # ax.set_xlabel('Date')
    ax.set_ylabel('Infectious')
    ax2.set_ylabel('Susceptible')
    ax2.set_ylim(0, 0.82)
    ax2.set_yticks(np.arange(0.4, 0.71, 0.1))
    ax2.set_yticklabels(["40%", "50%", "60%", "70%"])
    ax2.tick_params(axis='y', pad=1)

    ax.set_title("Impact of NPIs (environmental factor)", fontsize=7)
    ax.grid(axis='y', ls='--', zorder=0, alpha=0.3)
    ax.legend(frameon=False, ncols=2, fontsize=5,loc='lower left', bbox_to_anchor=(0, 0.75), columnspacing=0.8, handletextpad=0.5, handlelength=1.5,)
    # legend of infectious and susceptible
    legend_elements = [
        Line2D([0], [0], color='black', lw=1, linestyle='-', label='Infectious'),
        Line2D([0], [0], color='black', lw=1, linestyle='--', label='Susceptible')
    ]
    ax2.legend(handles=legend_elements, frameon=False, ncols=1, fontsize=5,loc='lower left', bbox_to_anchor=(0.45, 0.75), 
               columnspacing=0.8, handletextpad=0.5, handlelength=1.5)


    ax.axvspan(2019+40/52, 2021 + 8 / 12, facecolor='#FFEAC1', edgecolor='none', alpha=.6)
    ax.axvline(x=2021+11/12, zorder=0, color='#a9373b', ls='--', linewidth=1)
    # ax.text(x=2020.55, y=2500000, s='NPIs', c='#a9373b', fontsize=7)

    # C panel--by esp
    ax = fig.add_subplot(spec[2, 0])
    ax.text(-0.1, 0.95, 'c', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')

    ax.plot(df_v5.loc[:206, 'time_plot'], df_v5.loc[:206, "BV_y0.00_n0.00"]/N, linewidth=1, color='#2C357F', label='', 
            zorder=100, alpha=1.)
    ax.plot(df_v5.loc[:206, 'time_plot'], df_v5.loc[:206, "BY_y0.00_n0.00"]/N, linewidth=1, color='#AB0C1D', label='', 
            zorder=100, alpha=1.)

    ax.plot(df_v5.loc[206:, 'time_plot'], df_v5.loc[206:, "BV_y0.02_n0.10"]/N, linewidth=1, color='#2C357F', 
            alpha=0.6, label='BV (BY_esp=0.02)', zorder=100)
    ax.plot(df_v5.loc[206:, 'time_plot'], df_v5.loc[206:, "BY_y0.02_n0.10"]/N, linewidth=1, color='#AB0C1D', 
            alpha=0.6, label='BY (BY_esp=0.02)', zorder=100)
    ax.plot(df_v5.loc[206:, 'time_plot'], df_v5.loc[206:, "BV_y0.03_n0.10"]/N, linewidth=1, color='#2C357F', 
            alpha=0.3, label='BV (BY_esp=0.03)', zorder=100)
    ax.plot(df_v5.loc[206:, 'time_plot'], df_v5.loc[206:, "BY_y0.03_n0.10"]/N, linewidth=1, color='#AB0C1D', 
            alpha=0.3, label='BY (BY_esp=0.03)', zorder=100) 

    ax2 = ax.twinx()
    ax2.plot(df_v0.loc[:206, 'time_plot'], df_v0.loc[:206, ["S_y0.00_n0.00", "Ry_y0.00_n0.00"]].sum(axis=1)/N, linewidth=1, 
             alpha=1, color='#2C357F', label='BV', ls = '--', zorder=100)
    ax2.plot(df_v0.loc[:206, 'time_plot'], df_v0.loc[:206, ["S_y0.00_n0.00", "Rv_y0.00_n0.00"]].sum(axis=1)/N, linewidth=1, 
             alpha=1, color='#AB0C1D', label='BY', ls = '--', zorder=100)
    ax2.plot(df_v5.loc[206:, 'time_plot'], df_v5.loc[206:, ["S_y0.02_n0.10", "Ry_y0.02_n0.10"]].sum(axis=1)/N, linewidth=1, 
             alpha=0.6, color='#2C357F', label='BV', ls = '--', zorder=100)
    ax2.plot(df_v5.loc[206:, 'time_plot'], df_v5.loc[206:, ["S_y0.02_n0.10", "Rv_y0.02_n0.10"]].sum(axis=1)/N, linewidth=1, 
             alpha=0.6, color='#AB0C1D', label='BY', ls = '--', zorder=100)
    ax2.plot(df_v5.loc[206:, 'time_plot'], df_v5.loc[206:, ["S_y0.03_n0.10", "Ry_y0.03_n0.10"]].sum(axis=1)/N, linewidth=1, 
             alpha=0.3, color='#2C357F', label='BV', ls = '--', zorder=100)
    ax2.plot(df_v5.loc[206:, 'time_plot'], df_v5.loc[206:, ["S_y0.03_n0.10", "Rv_y0.03_n0.10"]].sum(axis=1)/N, linewidth=1, 
             alpha=0.3, color='#AB0C1D', label='BY', ls = '--', zorder=100)

    ax.set_xlim(2015.7, 2020+39/52)
    ax.set_xticks(np.arange(2016, 2021, 1))
    ax.set_ylim(0, 0.0215)
    ax.set_yticks(np.arange(0, 0.021, 0.005))
    ax.set_yticklabels(["0.0%", "0.5%", "1.0%", "1.5%", "2.0%"])
    ax.tick_params(axis='y', pad=1)
    # ax.set_xlabel('Date')
    ax.set_ylabel('Infectious')
    ax2.set_ylabel('Susceptible')
    ax2.set_ylim(0, 0.82)
    ax2.set_yticks(np.arange(0.4, 0.71, 0.1))
    ax2.set_yticklabels(["40%", "50%", "60%", "70%"])
    ax2.tick_params(axis='y', pad=1)
    
    # ax.set_xlabel('Date')
    ax.set_title("Impact of antigenicity (viral factor)", fontsize=7)
    ax.grid(axis='y', ls='--', zorder=0, alpha=0.3)
    ax.legend(frameon=False, ncols=2, fontsize=5,loc='lower left', bbox_to_anchor=(0, 0.75), columnspacing=0.8, handletextpad=0.5, handlelength=1.5,)
    # legend of infectious and susceptible
    legend_elements = [
        Line2D([0], [0], color='black', lw=1, linestyle='-', label='Infectious'),
        Line2D([0], [0], color='black', lw=1, linestyle='--', label='Susceptible')
    ]
    ax2.legend(handles=legend_elements, frameon=False, ncols=1, fontsize=5,loc='lower left', bbox_to_anchor=(0.5, 0.75), 
               columnspacing=0.8, handletextpad=0.5, handlelength=1.5)

    ax.axvspan(2019+40/52, 2021 + 8 / 12, facecolor='#FFEAC1', edgecolor='none', alpha=.6)
    ax.axvline(x=2021+11/12, zorder=0, color='#a9373b', ls='--', linewidth=1)
    # ax.text(x=2020.55, y=2500000, s='NPIs', c='#a9373b', fontsize=7)

    # D panel--by esp
    ax = fig.add_subplot(spec[3, 0])
    ax.text(-0.1, 0.95, 'd', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    
    ax.plot(df_r05.loc[:103, 'time_plot'], df_r05.loc[:103, "BV_y0.00_n0.10"]/N, linewidth=1, color='#2C357F', label='', 
            zorder=100, alpha=1.)
    ax.plot(df_r05.loc[:103, 'time_plot'], df_r05.loc[:103, "BY_y0.00_n0.10"]/N, linewidth=1, color='#AB0C1D', label='', 
            zorder=100, alpha=1.)

    ax.plot(df_r10.loc[102:, 'time_plot'], df_r10.loc[102:, "BV_y0.00_n0.10"]/N, linewidth=1, color='#2C357F', label='BV ($R_{0}^{BY}$=1.10)', 
            zorder=100, alpha=0.6)
    ax.plot(df_r10.loc[102:, 'time_plot'], df_r10.loc[102:, "BY_y0.00_n0.10"]/N, linewidth=1, color='#AB0C1D', label='BY ($R_{0}^{BY}$=1.10)', 
            zorder=100, alpha=0.6) 
    ax.plot(df_r05.loc[102:, 'time_plot'], df_r05.loc[102:, "BV_y0.00_n0.10"]/N, linewidth=1, color='#2C357F', label="BV ($R_{0}^{BY}$=1.05)", 
            zorder=100, alpha=0.3)
    ax.plot(df_r05.loc[102:, 'time_plot'], df_r05.loc[102:, "BY_y0.00_n0.10"]/N, linewidth=1, color='#AB0C1D', label='BY ($R_{0}^{BY}$=1.05)', 
            zorder=100, alpha=0.3)
    
    ax2 = ax.twinx()
    ax2.plot(df_r05.loc[:103,'time_plot'], df_r05.loc[:103, ["S_y0.00_n0.10", "Ry_y0.00_n0.10"]].sum(axis=1)/N, linewidth=1, 
             color='#2C357F', label='', ls = '--', zorder=100, alpha=1.)
    ax2.plot(df_r05.loc[:103,'time_plot'], df_r05.loc[:103, ["S_y0.00_n0.10", "Rv_y0.00_n0.10"]].sum(axis=1)/N, linewidth=1, 
             color='#AB0C1D', label='', ls = '--', zorder=100, alpha=1.)
    ax2.plot(df_r05.loc[102:,'time_plot'], df_r05.loc[102:, ["S_y0.00_n0.10", "Ry_y0.00_n0.10"]].sum(axis=1)/N, linewidth=1, 
             color='#2C357F', label='BV (R0=1.05)', ls = '--', zorder=100, alpha=0.3)
    ax2.plot(df_r05.loc[102:,'time_plot'], df_r05.loc[102:, ["S_y0.00_n0.10", "Rv_y0.00_n0.10"]].sum(axis=1)/N, linewidth=1, 
             color='#AB0C1D', label='BY (R0=1.05)', ls = '--', zorder=100, alpha=0.3)
    ax2.plot(df_r10.loc[102:,'time_plot'], df_r10.loc[102:, ["S_y0.00_n0.10", "Ry_y0.00_n0.10"]].sum(axis=1)/N, linewidth=1, 
             color='#2C357F', label='BV (R0=1.10)', ls = '--', zorder=100, alpha=0.6)
    ax2.plot(df_r10.loc[102:,'time_plot'], df_r10.loc[102:, ["S_y0.00_n0.10", "Rv_y0.00_n0.10"]].sum(axis=1)/N, linewidth=1, 
             color='#AB0C1D', label='BY (R0=1.10)', ls = '--', zorder=100, alpha=0.6)

    ax.set_xlim(2015.7, 2020+39/52)
    ax.set_ylim(0, 0.0215)
    ax.set_yticks(np.arange(0, 0.021, 0.005))
    ax.set_yticklabels(["0.0%", "0.5%", "1.0%", "1.5%", "2.0%"])
    ax.tick_params(axis='y', pad=1)
    # ax.set_xlabel('Date')
    ax.set_ylabel('Infectious')
    ax2.set_ylabel('Susceptible')
    ax2.set_ylim(0, 0.82)
    ax2.set_yticks(np.arange(0.4, 0.71, 0.1))
    ax2.set_yticklabels(["40%", "50%", "60%", "70%"])
    ax2.tick_params(axis='y', pad=1)
    ax.set_xlabel('Date')
    ax.set_title("Impact of 2017/2018 outbreak (host factor)", fontsize=7)
    ax.grid(axis='y', ls='--', zorder=0, alpha=0.3)
    
    leg1 = ax.legend(frameon=False, ncols=2, fontsize=5,loc='lower left', bbox_to_anchor=(0, 0.72), columnspacing=0.8, labelspacing=0.1, handletextpad=0.5, handlelength=1.5)
    leg1.set_zorder(200)
    # legend of infectious and susceptible
    legend_elements = [
        Line2D([0], [0], color='black', lw=1, linestyle='-', label='Infectious'),
        Line2D([0], [0], color='black', lw=1, linestyle='--', label='Susceptible')
    ]
    leg2 = ax2.legend(handles=legend_elements, frameon=False, ncols=1, fontsize=5,loc='lower left', bbox_to_anchor=(0.45, 0.75), 
               columnspacing=0.8, handletextpad=0.5, handlelength=1.5)
    leg2.set_zorder(200)

    ax.axvspan(2019+40/52, 2021 + 8 / 12, facecolor='#FFEAC1', edgecolor='none', alpha=.6)
    ax.axvline(x=2021+11/12, zorder=0, color='#a9373b', ls='--', linewidth=1)

    # E panel
    ax = fig.add_subplot(spec[0:2, 1])
    ax.text(-0.1, 0.98, 'e', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=13)
    heatmap = sns.heatmap(data=data_by_size_ne, annot=False,
                norm=norm,
                cbar_kws={'shrink': 1, 'aspect': 30, 'ticks':[0, 1, 3, 5, 7, 9, 11]},
                cmap="RdYlBu_r", square=False, ax=ax, alpha=1.)
    # Contour line
    X, Y = np.meshgrid(np.arange(data_by_size_ne.shape[1]), np.arange(data_by_size_ne.shape[0]))  # 生成坐标网格
    contour = ax.contour(X + 0.5, Y + 0.5, data_by_size_ne, levels=[0.5, 1, 4, 7, 10],
                         colors='black', linewidths=0.5, linestyles="-.")
    ax.clabel(contour, inline=True, fmt="%.1f", fontsize=6)

    ax.set_title('Impact of NPIs and antigenicity', fontsize=8)
    n_rows, n_cols = data_by_size_ne.shape
    ax.set_xticks(np.arange(0., n_cols, 20)+0.5, np.round(np.arange(0., 0.11, 0.02), 2))
    # ax.set_yticks(np.arange(0., n_rows, 20)+0.5, np.round(np.arange(0, 1.1, 0.2), 1))
    ax.set_yticks([0.5, 20.5, 35.7498234, 60.5, 80.5, 100.5], [0., 0.2, 0.35, 0.6, 0.8, 1.])
    colors = ['black', 'black', '#AB0C1D', 'black', 'black', 'black']
    # Traverse and set the color
    for label, color in zip(ax.get_yticklabels(), colors):
        label.set_color(color)
    ax.axhline(y=35.7498234, color='#AB0C1D', linestyle='--', linewidth=0.65)
    ax.invert_yaxis()
    ax.set_xlabel('Immune escape of B/Yamagata')
    ax.set_ylabel('NPIs')
    ax.tick_params(axis='y', rotation=0)
    ax.tick_params(axis='x', rotation=0)
    # colorbar label 
    cbar = heatmap.collections[0].colorbar
    cbar.ax.set_ylabel("Fold change of 2019/2020 epidemic size", rotation=90)


    # F panel
    ax = fig.add_subplot(spec[2:4, 1])
    ax.text(-0.1, 0.98, 'f', transform=ax.transAxes + trans, fontsize=10, weight='bold', va='bottom',
            fontfamily='Arial')
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=6)
    heatmap = sns.heatmap(data=data_by_size_nr, annot=False,
                norm=norm,
                cbar_kws={'shrink': 1, 'aspect': 30, 'ticks':[0, 1, 3, 5]},
                cmap="RdYlBu_r", square=False, ax=ax, alpha=1., linewidths=0)
    # Contour line
    X, Y = np.meshgrid(np.arange(data_by_size_nr.shape[1]), np.arange(data_by_size_nr.shape[0]))  # 生成坐标网格
    contour = ax.contour(X + 0.5, Y + 0.5, data_by_size_nr, levels=[0.5, 1, 2, 3, 4, 5],
                         colors='black', linewidths=0.5, linestyles="-.")
    ax.clabel(contour, inline=True, fmt="%.1f", fontsize=6)

    # ax.set_xlim(1, 52)
    ax.set_title('Epidemic size', fontsize=8)
    n_rows, n_cols = data_by_size_nr.shape
    ax.set_xticks(np.arange(0., n_cols, 20)+0.5, np.round(np.arange(1.0, 1.11, 0.02), 2))
    # ax.set_yticks(np.arange(0., n_rows, 20)+0.5, np.round(np.arange(0, 1.1, 0.2), 1))
    ax.set_yticks([0.5, 19.6195508, 40.5, 60.5, 80.5, 100.5], [0., 0.19, 0.4, 0.6, 0.8, 1.])
    colors = ['black', '#AB0C1D', 'black', 'black', 'black', 'black']
    # Traverse and set the color
    for label, color in zip(ax.get_yticklabels(), colors):
        label.set_color(color)
    ax.axhline(y=19.6195508, color='#AB0C1D', linestyle='--', linewidth=0.65)

    ax.invert_yaxis()
    ax.set_xlabel('$R_{0}^{BY}$ of 2017/2018-2018/2019 season')
    ax.set_ylabel('NPIs')
    ax.set_title('Impact of NPIs and 2017/2018 outbreak')
    ax.tick_params(axis='y', rotation=0)
    ax.tick_params(axis='x', rotation=0)
    # colorbar label 
    cbar = heatmap.collections[0].colorbar
    cbar.ax.set_ylabel("Fold change of 2019/2020 epidemic size", rotation=90)

fig.get_layout_engine().set(w_pad=10/300, h_pad=5/300, hspace=0.02, wspace=0.02)
# plt.savefig(r"./figure/paper_fig5_v9.pdf")
plt.savefig(r"./figure/fig5.png")