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
# cV                the strength of partial cross protection from BV to BY
# cY                the strength of partial cross protection from BY to BV
# PV                the duration of cross protection to BY
# PY                the duration of cross protection to BV
# omega             the infectious period difference between Ii and Ji


# In[3]:


def dz_dt(z, t, w_v, w_y, R0_v, R0_y, D, ksi, Lv, Ly, omega, cv, cy, immue_esp_pct_v1, immue_esp_pct_v2, immue_esp_pct_v3, immue_esp_pct_y1, immue_esp_pct_y2, npi_intensity, npi_start, npi_end, tv_0, ty_0, T, immue_esp_vt1, immue_esp_vt2, immue_esp_vt3, immue_esp_yt1, immue_esp_yt2, Pv, Py):
    
    """
    Two lineage transmission model equations. 
    """
    S, Iv, Iy, Rcv, Rcy, Rv, Ry, Jv, Jy, R, _, _ = z
    N = S + Iv + Iy + Rcv + Rcy + Rv + Ry + Jv + Jy + R

    # set import based on npi_intensity and epidemics
    # import_v = jnp.where((t >= npi_start) & (t <= npi_end), 1. * npi_intensity, 1.)
    # import_y = jnp.where((t >= npi_start) & (t <= npi_end), 1. * npi_intensity, 1.)
    import_v = 2.
    import_y = 2.

    # seasonal forced transmission rate
    sea_v = 1. + w_v * jnp.cos(2. * jnp.pi * (t - tv_0) / T)
    sea_y = 1. + w_y * jnp.cos(2. * jnp.pi * (t - ty_0) / T)

    # npi intensity
    # kappa = jnp.where((t >= npi_start) & (t <= npi_end), npi_intensity, 1)
    kappa = 1.

    # infection ability, 不加import?
    lambda_v = kappa * (R0_v / D) * sea_v * (Iv + ksi * Jv) / N
    lambda_y = kappa * (R0_y / D) * sea_y * (Iy + ksi * Jy) / N

    # escape population
    esp_pct_v1 = jnp.where((jnp.abs(t - immue_esp_vt1) <= 27.), immue_esp_pct_v1, 0)
    esp_pct_v2 = jnp.where((jnp.abs(t - immue_esp_vt2) <= 27.), immue_esp_pct_v2, 0)
    esp_pct_v3 = jnp.where((jnp.abs(t - immue_esp_vt3) <= 27.), immue_esp_pct_v3, 0)
    esp_pct_y1 = jnp.where((jnp.abs(t - immue_esp_yt1) <= 27.), immue_esp_pct_y1, 0)
    esp_pct_y2 = jnp.where((jnp.abs(t - immue_esp_yt2) <= 27.), immue_esp_pct_y2, 0)
    
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
data = pd.read_csv(r"../../../data/surveillance/flu_incidence_us.csv", index_col = 0)
data['Victoria_IR'] = data['ILI%'] * data['positive_rate'] * data['Victoria_proporation'] * 100000
data['Yamagata_IR'] = data['ILI%'] * data['positive_rate'] * data['Yamagata_proporation'] * 100000
data['Victoria_IR_rolling'] = data['Victoria_IR'].rolling(window=4).mean().round()
data['Yamagata_IR_rolling'] = data['Yamagata_IR'].rolling(window=4).mean().round()
data = data.dropna(subset=['Victoria_IR_rolling','Yamagata_IR_rolling'])
data = data.iloc[:153, :].copy()
data.reset_index(inplace=True, drop=True)
data[['Victoria_IR_rolling','Yamagata_IR_rolling']] = data[['Victoria_IR_rolling','Yamagata_IR_rolling']].replace(0, 1e-99)
data


# In[5]:


# Define cases for MCMC
y_v = jnp.array(data['Victoria_IR_rolling'].values)
y_y = jnp.array(data['Yamagata_IR_rolling'].values)
y = [y_v, y_y]


# In[6]:


def step_time(step):
    
    '''transfer step to time'''

    step_time = 2015 + (43 / 52) + ((step-1) / 52)
    
    return step_time


# In[7]:


steps = y_v.shape[0]


# In[8]:


def generate_data(S_rate_ini, Iv_rate_ini, Iy_rate_ini, Rcv_rate_ini, Rcy_rate_ini,
                  Rv_rate_ini, Ry_rate_ini, Jv_rate_ini, Jy_rate_ini, R_rate_ini,
                  R0_v, R0_y, w_v, w_y, cv, cy, tv_0, ty_0, immue_esp_pct_v1, immue_esp_pct_v2, immue_esp_pct_v3, 
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
    z = odeint(dz_dt, z_init, ts, w_v, w_y, R0_v, R0_y, D, ksi, Lv, Ly, omega, cv, cy, immue_esp_pct_v1, immue_esp_pct_v2, immue_esp_pct_v3, immue_esp_pct_y1, immue_esp_pct_y2, npi_intensity, npi_start, npi_end, tv_0, ty_0, T, immue_esp_vt1, immue_esp_vt2, immue_esp_vt3, immue_esp_yt1, immue_esp_yt2, Pv, Py, rtol=1e-6, atol=1e-5, mxstep=1000)

    return z


# In[9]:


def params_estimate(S_rate_ini, Iv_rate_ini, Iy_rate_ini, Rcv_rate_ini, Rcy_rate_ini,
                    Rv_rate_ini, Ry_rate_ini, Jv_rate_ini, Jy_rate_ini, R_rate_ini,
                    R0_v, R0_y, w_v, w_y, cv, cy, tv_0, ty_0, det_prob_v, det_prob_y,
                    immue_esp_pct_v1, immue_esp_pct_v2, immue_esp_pct_v3, 
                    immue_esp_pct_y1, steps, npi_intensity, file_name):
    
    ss = generate_data(S_rate_ini = S_rate_ini, 
                       Iv_rate_ini = Iv_rate_ini, Iy_rate_ini = Iy_rate_ini, 
                       Rcv_rate_ini = Rcv_rate_ini, Rcy_rate_ini = Rcy_rate_ini,
                       Rv_rate_ini = Rv_rate_ini, Ry_rate_ini = Ry_rate_ini, 
                       Jv_rate_ini = Jv_rate_ini, Jy_rate_ini = Jy_rate_ini, 
                       R_rate_ini = R_rate_ini,
                       R0_v = R0_v, R0_y = R0_y, 
                       w_v = w_v, w_y = w_y, 
                       cv = cv, cy = cy, 
                       tv_0 = tv_0, ty_0 = ty_0,
                       immue_esp_pct_v1 = immue_esp_pct_v1, 
                       immue_esp_pct_v2 = immue_esp_pct_v2,
                       immue_esp_pct_v3 = immue_esp_pct_v3,
                       immue_esp_pct_y1 = immue_esp_pct_y1,
                       npi_intensity = npi_intensity,
                       steps = steps)
    # new cases
    inf_hat_v = det_prob_v * jnp.diff(ss[:,10])
    inf_hat_y = det_prob_y * jnp.diff(ss[:,11])

    df_ci = pd.DataFrame({f'inf_hat_v_{file_name}': inf_hat_v, f'inf_hat_y_{file_name}': inf_hat_y})

    df_ci.to_csv(f"./result/fluB_ci/data/case_{file_name}.csv", index=False)


# In[10]:


# import estimated params
df = pd.read_csv(r'../../../data/result/dynamic_model/B_params_rank.csv', index_col=0)


# In[11]:


list_params = ['R0_v', 'R0_y', 'w_v', 'w_y', 'cv', 'cy', 'tv_0', 'ty_0', 'det_prob_v', 'det_prob_y']
list_state = ['S_rate_ini', 'Rcv_rate_ini', 'Rcy_rate_ini', 'Rv_rate_ini', 
             'Ry_rate_ini', 'R_rate_ini', 'Iv_rate_ini', 'Iy_rate_ini', 'Jv_rate_ini', 'Jy_rate_ini']


# In[12]:


dict_params = {}


# In[13]:


# determine the value of initial compartment BV and BY
for param in list_state:
    param_mean = param + '_adjust_mean'
    dict_params[param] = df.loc[0, param_mean]


# In[14]:


# determine the value of params BV and BY
for param in list_params:
    param_mean = param + '_mean'
    dict_params[param] = df.loc[0, param_mean]


# In[15]:


dict_params


# In[16]:


# params_estimate(S_rate_ini = dict_params['S_rate_ini'], Iv_rate_ini = dict_params['Iv_rate_ini'], 
#                 Iy_rate_ini = dict_params['Iy_rate_ini'], 
#                 Rcv_rate_ini = dict_params['Rcv_rate_ini'], Rcy_rate_ini = dict_params['Rcy_rate_ini'],
#                 Rv_rate_ini = dict_params['Rv_rate_ini'], Ry_rate_ini = dict_params['Ry_rate_ini'], 
#                 Jv_rate_ini = dict_params['Jv_rate_ini'], Jy_rate_ini = dict_params['Jy_rate_ini'], 
#                 R_rate_ini = dict_params['R_rate_ini'],
#                 R0_v = dict_params['R0_v'], R0_y = dict_params['R0_y'],
#                 w_v = 0.10110991, w_y = 0.33759985, 
#                 cv = 0.12310178, cy = 0.3098479, tv_0 = 16.00632857, ty_0 = 13.99186675, 
#                 r = 0.46940369, file_name = 'test',
#                 immue_esp_pct_v1 = 0., immue_esp_pct_v2 = 0., immue_esp_pct_v3 = 0., immue_esp_pct_y1 = 0.0,
#                 npi_intensity = 0., steps=y_v.shape[0])


# In[17]:


params_estimate(S_rate_ini = dict_params['S_rate_ini'], Iv_rate_ini = dict_params['Iv_rate_ini'], 
                Iy_rate_ini = dict_params['Iy_rate_ini'], 
                Rcv_rate_ini = dict_params['Rcv_rate_ini'], Rcy_rate_ini = dict_params['Rcy_rate_ini'],
                Rv_rate_ini = dict_params['Rv_rate_ini'], Ry_rate_ini = dict_params['Ry_rate_ini'], 
                Jv_rate_ini = dict_params['Jv_rate_ini'], Jy_rate_ini = dict_params['Jy_rate_ini'], 
                R_rate_ini = dict_params['R_rate_ini'],
                R0_v = float(sys.argv[1]), R0_y = float(sys.argv[2]),
                w_v = float(sys.argv[3]), w_y = float(sys.argv[4]), 
                cv = float(sys.argv[5]), cy = float(sys.argv[6]), tv_0 = float(sys.argv[7]), ty_0 = float(sys.argv[8]), 
                det_prob_v = float(sys.argv[9]), det_prob_y = float(sys.argv[10]), file_name = sys.argv[11],
                immue_esp_pct_v1 = 0., immue_esp_pct_v2 = 0., immue_esp_pct_v3 = 0., immue_esp_pct_y1 = 0.0,
                npi_intensity = 0., steps=y_v.shape[0]+55)

