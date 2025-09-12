#!/usr/bin/env python
# coding: utf-8

# In[1]:


import functools
import numpy as np
import matplotlib.pyplot as plt
import time
import jax
from jax.experimental.ode import odeint
import jax.numpy as jnp
from jax.random import PRNGKey
import sys
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_sample
from jax import random, jit, lax
from numpyro.distributions import constraints, Distribution
from numpyro.distributions.util import validate_sample
from jax.scipy.special import gammaln
import argparse
import arviz as az
import os
import pandas as pd
import datetime
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error

numpyro.enable_x64(True)
numpyro.set_platform("cpu")
numpyro.set_host_device_count(1)


# ### Model

# In[2]:


# Compartment       Definition
# S                 Susceptible individuals with no prior immunity
# Iv                Infectious with BV; no prior immunity
# Iy                Infectious with BY; no prior immunity
# Rcv               Recoverd from BV; cross protected against BY
# Rcy               Recoverd from BY; cross protected against BV
# Rv                Recoverd from BV; no cross protection
# Ry                Recoverd from BY; no cross protection
# Jy                Infectious with BY; prior immunity to BV
# Jv                Infectious with BV; prior immunity to BY
# R                 Recoverd from BV and BY

# Parameters        Definition
# Nt                Populatin size
# mu                Natural birth and death rate
# beta              transmission rate
# ksi               the relative transmission difference between Ii and Ji
# alpha             the heterogeneous mixing effects on the spread of pathogen in the population
# D                 infectious period
# LV                Duration of BV specific immunity
# LY                Duration of BY specific immunity
# cv                the strength of partial cross protection from BV to BY
# cv                the strength of partial cross protection from BY to BV
# PV                the duration of cross protection to BY
# PY                the duration of cross protection to BV
# omega             the infectious period difference between Ii and Ji


# In[3]:


def dz_dt(z, t, tv_c, ty_c, w_v, w_y, R0_v, R0_y, D, ksi, Lv, Ly, omega, cv, cy, immue_esp_pct_v1, immue_esp_pct_v2, immue_esp_pct_v3, immue_esp_pct_y1, immue_esp_pct_y2, npi_intensity, npi_start, npi_end, tv_0_old, tv_0_new, ty_0_old, ty_0_new, T, immue_esp_vt1, immue_esp_vt2, immue_esp_vt3, immue_esp_yt1, immue_esp_yt2, Pv, Py):
    
    """
    Two lineage transmission model equations. 
    """
    S, Iv, Iy, Rcv, Rcy, Rv, Ry, Jv, Jy, R, _, _ = z
    N = S + Iv + Iy + Rcv + Rcy + Rv + Ry + Jv + Jy + R
 
    # set import based on npi_intensity and epidemics
    # import_v = jnp.where((t >= npi_start) & (t <= npi_end), 0., 2.)
    # import_y = jnp.where((t >= npi_start) & (t <= npi_end), 0., 2.)
    import_v = jnp.where(
        (t >= npi_start) & (t <= npi_end),
        jnp.where(npi_intensity == 0., 2., 0.),
        2.
    )
    
    import_y = jnp.where(
        (t >= npi_start) & (t <= npi_end),
        jnp.where(npi_intensity == 0., 2., 0.),
        2.
    )

    # cross protection
    tv_0 = jnp.where((t >= tv_c), tv_0_new, tv_0_old)
    ty_0 = jnp.where((t >= ty_c), ty_0_new, ty_0_old)

    # seasonal forced transmission rate
    sea_v = 1. + w_v * jnp.cos(2. * jnp.pi * (t - tv_0) / T)
    sea_y = 1. + w_y * jnp.cos(2. * jnp.pi * (t - ty_0) / T)

    # npi intensity
    kappa = jnp.where((t >= npi_start) & (t <= npi_end), 1-npi_intensity, 1)

    # infection ability
    lambda_v = kappa * (R0_v / D) * sea_v * (Iv + ksi * Jv) / N
    lambda_y = kappa * (R0_y / D) * sea_y * (Iy + ksi * Jy) / N

    # escape population
    esp_pct_v1 = jnp.where(((t - immue_esp_vt1) <= 27.) & ((t - immue_esp_vt1) >= 0.), immue_esp_pct_v1, 0)
    esp_pct_v2 = jnp.where(((t - immue_esp_vt2) <= 27.) & ((t - immue_esp_vt2) >= 0.), immue_esp_pct_v2, 0)
    esp_pct_v3 = jnp.where(((t - immue_esp_vt3) <= 27.) & ((t - immue_esp_vt3) >= 0.), immue_esp_pct_v3, 0)
    esp_pct_y1 = jnp.where(((t - immue_esp_yt1) <= 27.) & ((t - immue_esp_yt1) >= 0.), immue_esp_pct_y1, 0)
    esp_pct_y2 = jnp.where(((t - immue_esp_yt2) <= 27.) & ((t - immue_esp_yt2) >= 0.), immue_esp_pct_y2, 0)

    # differential equations
    dS_dt = -lambda_v * S - lambda_y * S + (Rv / Lv) + (Ry / Ly) + esp_pct_v1 * Rv + esp_pct_v2 * Rv + esp_pct_v3 * Rv + esp_pct_y1 * Ry + esp_pct_y2 * Ry - import_v - import_y 
    dIv_dt = lambda_v * S - (Iv / D) + import_v
    dIy_dt = lambda_y * S - (Iy / D) + import_y
    dRcv_dt = (Iv / D) - (Rcv / Pv) - cv * lambda_y * Rcv
    dRcy_dt = (Iy / D) - (Rcy / Py) - cy * lambda_v * Rcy
    dRv_dt = (Rcv / Pv) + (R / Ly) - (Rv / Lv) - lambda_y * Rv - esp_pct_v1 * Rv - esp_pct_v2 * Rv - esp_pct_v3 * Rv + esp_pct_y1 * R + esp_pct_y2 * R
    dRy_dt = (Rcy / Py) + (R / Lv) - (Ry / Ly) - lambda_v * Ry - esp_pct_y1 * Ry - esp_pct_y2 * Ry + esp_pct_v1 * R + esp_pct_v2 * R + esp_pct_v3 * R
    dJv_dt = cy * lambda_v * Rcy + lambda_v * Ry - (Jv / (omega * D))
    dJy_dt = cv * lambda_y * Rcv + lambda_y * Rv - (Jy / (omega * D))
    dR_dt = (Jy / (omega * D)) + (Jv / (omega * D)) -(R / Ly) - (R / Lv) - esp_pct_v1 * R - esp_pct_v2 * R - esp_pct_v3 * R - esp_pct_y1 * R - esp_pct_y2 * R

    # auxiliary variable -- new cases
    dVnew_dt = lambda_v * S + import_v + cy * lambda_v * Rcy + lambda_v * Ry
    dYnew_dt = lambda_y * S + import_y + cv * lambda_y * Rcv + lambda_y * Rv
    
    return jnp.stack([dS_dt, dIv_dt, dIy_dt, dRcv_dt, dRcy_dt, dRv_dt, dRy_dt, dJv_dt, dJy_dt, dR_dt, dVnew_dt, dYnew_dt])


# In[4]:


# import surveillence data
data = pd.read_csv(r"../../../data/flu_incidence_us.csv", index_col = 0)
data['Victoria_IR'] = data['ILI%'] * data['positive_rate'] * data['Victoria_proporation'] * 100000
data['Yamagata_IR'] = data['ILI%'] * data['positive_rate'] * data['Yamagata_proporation'] * 100000
data['Victoria_IR_rolling'] = data['Victoria_IR'].rolling(window=4).mean().round()
data['Yamagata_IR_rolling'] = data['Yamagata_IR'].rolling(window=4).mean().round()
data = data.dropna(subset=['Victoria_IR_rolling','Yamagata_IR_rolling'])
# data = data.iloc[:, :].copy()
data.reset_index(inplace=True, drop=True)
data[['Victoria_IR_rolling','Yamagata_IR_rolling']] = data[['Victoria_IR_rolling','Yamagata_IR_rolling']].replace(0, 1e-99)
data


# In[5]:


data.query('year_week=="2019-40"').index


# In[6]:


# Define cases for MCMC
y_v = jnp.array(data['Victoria_IR_rolling'].values)
y_y = jnp.array(data['Yamagata_IR_rolling'].values)
y = jnp.array([y_v, y_y])


# In[7]:


def step_time(step):
    
    '''transfer step to time'''

    step_time = 2015 + (43 / 52) + ((step-1) / 52)
    
    return step_time


# In[8]:


def generate_data(S_rate_ini, Iv_rate_ini, Iy_rate_ini, Rcv_rate_ini, Rcy_rate_ini,
                  Rv_rate_ini, Ry_rate_ini, Jv_rate_ini, Jy_rate_ini, R_rate_ini,
                  R0_v, R0_y, w_v, w_y, cv, cy, tv_0_old, tv_0_new, ty_0_old, ty_0_new, tv_c, ty_c, immue_esp_pct_v1, immue_esp_pct_v2, immue_esp_pct_v3, 
                  immue_esp_pct_y1, steps, npi_intensity):

    # steps
    M = steps
    
    # initial state proportions, not treated as parameters
    N=100000.

    S_ini   = N * S_rate_ini  
    Iv_ini  = N * Iv_rate_ini  
    Iy_ini  = N * Iy_rate_ini  
    Rcv_ini = N * Rcv_rate_ini 
    Rcy_ini = N * Rcy_rate_ini 
    Rv_ini  = N * Rv_rate_ini  
    Ry_ini  = N * Ry_rate_ini  
    Jv_ini  = N * Jv_rate_ini  
    Jy_ini  = N * Jy_rate_ini  
    R_ini   = N * R_rate_ini   
    z_init = jnp.array([S_ini, Iv_ini, Iy_ini, Rcv_ini, Rcy_ini, Rv_ini, Ry_ini, Jv_ini, Jy_ini, R_ini, 0.0, 0.0])


    # measurement times
    ts = jnp.arange(float(M+1))

    # R0_v = 1.3
    # R0_y = 1.2
    D  = 3.4 / 7.
    ksi = 1.
    Lv = 4. * 52
    Ly = 4. * 52
    Pv = (1./12) * 52
    Py = (1./12) * 52
    omega = 1.

    # fixed parameters
    mu = float(0.0098/52)
    npi_start = 218.
    npi_end = 301.
    # npi_intensity = 1.
    T = float(52.)
    immue_esp_vt1 = float(101)
    immue_esp_vt2 = float(205)
    immue_esp_vt3 = float(309)
    immue_esp_yt1 = float(205)
    immue_esp_yt2 = float(260) 
    # immue_esp_pct_v3 = 0.
    # immue_esp_pct_y1 = 0.
    immue_esp_pct_y2 = 0.

    # integrate dz/dt, z will have shape M x 10
    z = odeint(dz_dt, z_init, ts, tv_c, ty_c, w_v, w_y, R0_v, R0_y, D, ksi, Lv, Ly, omega, cv, cy, immue_esp_pct_v1, immue_esp_pct_v2, immue_esp_pct_v3, immue_esp_pct_y1, immue_esp_pct_y2, npi_intensity, npi_start, npi_end, tv_0_old, tv_0_new, ty_0_old, ty_0_new, T, immue_esp_vt1, immue_esp_vt2, immue_esp_vt3, immue_esp_yt1, immue_esp_yt2, Pv, Py, rtol=1e-6, atol=1e-5, mxstep=1000)

    return z


# In[9]:


# import estimated params
df = pd.read_csv(r'./B_params_rank.csv', index_col=0)


# In[10]:


list_params = ['R0_v', 'R0_y', 'w_v', 'w_y', 'cv', 'cy', 'tv_0', 'ty_0', 'det_prob_v', 'det_prob_y']
list_state = ['S_rate_ini', 'Rcv_rate_ini', 'Rcy_rate_ini', 'Rv_rate_ini', 
             'Ry_rate_ini', 'R_rate_ini', 'Iv_rate_ini', 'Iy_rate_ini', 'Jv_rate_ini', 'Jy_rate_ini']


# In[11]:


dict_params = {}


# In[12]:


# determine the value of initial compartment BV and BY
for param in list_state:
    param_mean = param + '_adjust_mean'
    dict_params[param] = df.loc[0, param_mean]


# In[13]:


# determine the value of params BV and BY
for param in list_params:
    param_mean = param + '_mean'
    dict_params[param] = df.loc[0, param_mean]


# In[14]:


# In[15]:


# generate the simmulation result
bv_esp = 0.05
by_esp = 0.00
npi = 0.10
for cv_value in tqdm(np.arange(0.0, 30.1, 0.1)):
    for cy_value in np.arange(0.0, 30.1, 0.1):
        ss = generate_data(S_rate_ini = dict_params['S_rate_ini'], Iv_rate_ini = dict_params['Iv_rate_ini'], 
                           Iy_rate_ini = dict_params['Iy_rate_ini'], 
                           Rcv_rate_ini = dict_params['Rcv_rate_ini'], Rcy_rate_ini = dict_params['Rcy_rate_ini'],
                           Rv_rate_ini = dict_params['Rv_rate_ini'], Ry_rate_ini = dict_params['Ry_rate_ini'], 
                           Jv_rate_ini = dict_params['Jv_rate_ini'], Jy_rate_ini = dict_params['Jy_rate_ini'], 
                           R_rate_ini = dict_params['R_rate_ini'],
                           R0_v = dict_params['R0_v'], R0_y = dict_params['R0_y'],
                           w_v = dict_params['w_v'], w_y = dict_params['w_y'], 
                           tv_0_old = dict_params['tv_0'], tv_0_new = cv_value, 
                           ty_0_old = dict_params['ty_0'], ty_0_new = cy_value, 
                           cv = dict_params['cv'], cy = dict_params['cy'], tv_c = 206, ty_c = 206,
                           immue_esp_pct_v1 = 0., immue_esp_pct_v2 = bv_esp, immue_esp_pct_v3 = 0., 
                           immue_esp_pct_y1 = by_esp, steps = y_v.shape[0]+52*5, npi_intensity = npi)
        # S, Iv, Iy, Rcv, Rcy, Rv, Ry, Jv, Jy, R, _, _
        S = ss[:,0]
        Iv =ss[:,1]
        Iy = ss[:,2]
        Rcv = ss[:,3]
        Rcy = ss[:,4]
        Rv = ss[:,5]
        Ry = ss[:,6]
        Jv = ss[:,7]
        Jy = ss[:,8]
        R = ss[:,9]
        CV = ss[:,10]
        CY = ss[:,11]
        df_result = pd.DataFrame()
        df_result[f'S_y_cv{cv_value:.2f}_cy{cy_value:.2f}'] = S
        df_result[f'Iv_y_cv{cv_value:.2f}_cy{cy_value:.2f}'] = Iv
        df_result[f'Iy_y_cv{cv_value:.2f}_cy{cy_value:.2f}'] = Iy
        df_result[f'Rcv_y_cv{cv_value:.2f}_cy{cy_value:.2f}'] = Rcv
        df_result[f'Rcy_y_cv{cv_value:.2f}_cy{cy_value:.2f}'] = Rcy
        df_result[f'Rv_y_cv{cv_value:.2f}_cy{cy_value:.2f}'] = Rv
        df_result[f'Ry_y_cv{cv_value:.2f}_cy{cy_value:.2f}'] = Ry
        df_result[f'Jv_y_cv{cv_value:.2f}_cy{cy_value:.2f}'] = Jv
        df_result[f'Jy_y_cv{cv_value:.2f}_cy{cy_value:.2f}'] = Jy
        df_result[f'R_y_cv{cv_value:.2f}_cy{cy_value:.2f}'] = R
        df_result[f'CV_y_cv{cv_value:.2f}_cy{cy_value:.2f}'] = CV
        df_result[f'CY_y_cv{cv_value:.2f}_cy{cy_value:.2f}'] = CY
        df_result[f'BV_y_cv{cv_value:.2f}_cy{cy_value:.2f}'] = dict_params['det_prob_v'] * df_result[f'CV_y_cv{cv_value:.2f}_cy{cy_value:.2f}'].diff()
        df_result[f'BY_y_cv{cv_value:.2f}_cy{cy_value:.2f}'] = dict_params['det_prob_y'] * df_result[f'CY_y_cv{cv_value:.2f}_cy{cy_value:.2f}'].diff()
        
        df_result['step'] = np.arange(len(df_result))
        df_result['time_plot'] = df_result.apply(lambda row: step_time(row['step']), axis=1)
        df_result.set_index('time_plot', inplace=True)
        df_result.to_csv(f"./result/simmulation/data_t/data_simmulation_orign_cv{cv_value:.2f}_cy{cy_value:.2f}.csv") 


# In[16]:


# # generate the simmulation result
# for bv_esp in tqdm(np.arange(0.10, 0.11, 0.01)):
#     list_df = [] # avoid warning of dataframe for inserting too many columns
#     for by_esp in np.arange(0.000, 0.001, 0.001):
#         for npi in np.arange(0.15, 0.16, 0.01):
#             # for cv_value in np.arange(0.0, 1.01, 0.01):
#             #     for cy_value in np.arange(0.0, 1.01, 0.01):
#             for cv_value in [0.136, 0.285]:
#                 for cy_value in [0.136, 0.285]:
#                     ss = generate_data(S_rate_ini = dict_params['S_rate_ini'], Iv_rate_ini = dict_params['Iv_rate_ini'], 
#                                        Iy_rate_ini = dict_params['Iy_rate_ini'], 
#                                        Rcv_rate_ini = dict_params['Rcv_rate_ini'], Rcy_rate_ini = dict_params['Rcy_rate_ini'],
#                                        Rv_rate_ini = dict_params['Rv_rate_ini'], Ry_rate_ini = dict_params['Ry_rate_ini'], 
#                                        Jv_rate_ini = dict_params['Jv_rate_ini'], Jy_rate_ini = dict_params['Jy_rate_ini'], 
#                                        R_rate_ini = dict_params['R_rate_ini'],
#                                        R0_v = dict_params['R0_v'], R0_y = dict_params['R0_y'],
#                                        w_v = dict_params['w_v'], w_y = dict_params['w_y'], 
#                                        cv_old = dict_params['cv'], cv_new = cv_value, 
#                                        cy_old = dict_params['cy'], cy_new = cy_value, 
#                                        tv_0 = dict_params['tv_0'], ty_0 = dict_params['ty_0'], tv_c = 206, ty_c = 206,
#                                        immue_esp_pct_v1 = 0., immue_esp_pct_v2 = bv_esp, immue_esp_pct_v3 = 0., 
#                                        immue_esp_pct_y1 = by_esp, steps = y_v.shape[0]+52*5, npi_intensity = npi)
#                     # S, Iv, Iy, Rcv, Rcy, Rv, Ry, Jv, Jy, R, _, _
#                     S = ss[:,0]
#                     Iv =ss[:,1]
#                     Iy = ss[:,2]
#                     Rcv = ss[:,3]
#                     Rcy = ss[:,4]
#                     Rv = ss[:,5]
#                     Ry = ss[:,6]
#                     Jv = ss[:,7]
#                     Jy = ss[:,8]
#                     R = ss[:,9]
#                     CV = ss[:,10]
#                     CY = ss[:,11]
#                     df_result = pd.DataFrame()
#                     df_result[f'S_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.2f}_cy{cy_value:.2f}'] = S
#                     df_result[f'Iv_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.2f}_cy{cy_value:.2f}'] = Iv
#                     df_result[f'Iy_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.2f}_cy{cy_value:.2f}'] = Iy
#                     df_result[f'Rcv_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.2f}_cy{cy_value:.2f}'] = Rcv
#                     df_result[f'Rcy_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.2f}_cy{cy_value:.2f}'] = Rcy
#                     df_result[f'Rv_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.2f}_cy{cy_value:.2f}'] = Rv
#                     df_result[f'Ry_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.2f}_cy{cy_value:.2f}'] = Ry
#                     df_result[f'Jv_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.2f}_cy{cy_value:.2f}'] = Jv
#                     df_result[f'Jy_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.2f}_cy{cy_value:.2f}'] = Jy
#                     df_result[f'R_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.2f}_cy{cy_value:.2f}'] = R
#                     df_result[f'CV_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.2f}_cy{cy_value:.2f}'] = CV
#                     df_result[f'CY_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.2f}_cy{cy_value:.2f}'] = CY
#                     df_result[f'BV_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.2f}_cy{cy_value:.2f}'] = dict_params['r'] * df_result[f'CV_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.2f}_cy{cy_value:.2f}'].diff()
#                     df_result[f'BY_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.2f}_cy{cy_value:.2f}'] = dict_params['r'] * df_result[f'CY_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.2f}_cy{cy_value:.2f}'].diff()
                
#                     list_df.append(df_result)
#     # merge all dataframes
#     df_final = pd.concat(list_df, axis=1).copy()
#     df_final['step'] = np.arange(len(df_final))
#     df_final['time_plot'] = df_final.apply(lambda row: step_time(row['step']), axis=1)
#     df_final.set_index('time_plot', inplace=True)
#     df_final.to_csv(f"./data/change_c/data_simmulation_orign_v{bv_esp:.2f}.csv")


# In[17]:


# # generate the simmulation result
# for bv_esp in tqdm(np.arange(0.10, 0.11, 0.01)):
#     list_df = [] # avoid warning of dataframe for inserting too many columns
#     for by_esp in np.arange(0.000, 0.001, 0.001):
#         for npi in [0.00, 0.15]:
#             for cv_value in [0.000, 0.136, 0.285]:
#                 for cy_value in [0.000, 0.136, 0.285]:
#                     ss = generate_data(S_rate_ini = dict_params['S_rate_ini'], Iv_rate_ini = dict_params['Iv_rate_ini'], 
#                                        Iy_rate_ini = dict_params['Iy_rate_ini'], 
#                                        Rcv_rate_ini = dict_params['Rcv_rate_ini'], Rcy_rate_ini = dict_params['Rcy_rate_ini'],
#                                        Rv_rate_ini = dict_params['Rv_rate_ini'], Ry_rate_ini = dict_params['Ry_rate_ini'], 
#                                        Jv_rate_ini = dict_params['Jv_rate_ini'], Jy_rate_ini = dict_params['Jy_rate_ini'], 
#                                        R_rate_ini = dict_params['R_rate_ini'],
#                                        R0_v = dict_params['R0_v'], R0_y = dict_params['R0_y'],
#                                        w_v = dict_params['w_v'], w_y = dict_params['w_y'], 
#                                        cv_old = dict_params['cv'], cv_new = cv_value, 
#                                        cy_old = dict_params['cy'], cy_new = cy_value, 
#                                        tv_0 = dict_params['tv_0'], ty_0 = dict_params['ty_0'], tv_c = 206, ty_c = 206,
#                                        immue_esp_pct_v1 = 0., immue_esp_pct_v2 = bv_esp, immue_esp_pct_v3 = 0., 
#                                        immue_esp_pct_y1 = by_esp, steps = y_v.shape[0]+52*5, npi_intensity = npi)
#                     # S, Iv, Iy, Rcv, Rcy, Rv, Ry, Jv, Jy, R, _, _
#                     S = ss[:,0]
#                     Iv =ss[:,1]
#                     Iy = ss[:,2]
#                     Rcv = ss[:,3]
#                     Rcy = ss[:,4]
#                     Rv = ss[:,5]
#                     Ry = ss[:,6]
#                     Jv = ss[:,7]
#                     Jy = ss[:,8]
#                     R = ss[:,9]
#                     CV = ss[:,10]
#                     CY = ss[:,11]
#                     df_result = pd.DataFrame()
#                     df_result[f'S_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.3f}_cy{cy_value:.3f}'] = S
#                     df_result[f'Iv_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.3f}_cy{cy_value:.3f}'] = Iv
#                     df_result[f'Iy_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.3f}_cy{cy_value:.3f}'] = Iy
#                     df_result[f'Rcv_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.3f}_cy{cy_value:.3f}'] = Rcv
#                     df_result[f'Rcy_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.3f}_cy{cy_value:.3f}'] = Rcy
#                     df_result[f'Rv_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.3f}_cy{cy_value:.3f}'] = Rv
#                     df_result[f'Ry_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.3f}_cy{cy_value:.3f}'] = Ry
#                     df_result[f'Jv_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.3f}_cy{cy_value:.3f}'] = Jv
#                     df_result[f'Jy_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.3f}_cy{cy_value:.3f}'] = Jy
#                     df_result[f'R_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.3f}_cy{cy_value:.3f}'] = R
#                     df_result[f'CV_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.3f}_cy{cy_value:.3f}'] = CV
#                     df_result[f'CY_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.3f}_cy{cy_value:.3f}'] = CY
#                     df_result[f'BV_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.3f}_cy{cy_value:.3f}'] = dict_params['r'] * df_result[f'CV_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.3f}_cy{cy_value:.3f}'].diff()
#                     df_result[f'BY_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.3f}_cy{cy_value:.3f}'] = dict_params['r'] * df_result[f'CY_y{by_esp:.3f}_n{npi:.2f}_cv{cv_value:.3f}_cy{cy_value:.3f}'].diff()
                
#                     list_df.append(df_result)
#     # merge all dataframes
#     df_final = pd.concat(list_df, axis=1).copy()
#     df_final['step'] = np.arange(len(df_final))
#     df_final['time_plot'] = df_final.apply(lambda row: step_time(row['step']), axis=1)
#     df_final.set_index('time_plot', inplace=True)
#     df_final.to_csv(f"./data/change_c/data_simmulation_orign_v{bv_esp:.2f}_example.csv")


# In[ ]:




