#!/usr/bin/env python
# coding: utf-8

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
# Compartment       Definition
'''We will upload the complete code here once the manuscript is officially published'''

def dz_dt(z, t, w_v, w_y, R0_v, R0_y, D, ksi, Lv, Ly, omega, cv, cy, immue_esp_pct_v1, immue_esp_pct_v2, immue_esp_pct_v3, immue_esp_pct_y1, immue_esp_pct_y2, npi_intensity, npi_start, npi_end, tv_0, ty_0, T, immue_esp_vt1, immue_esp_vt2, immue_esp_vt3, immue_esp_yt1, immue_esp_yt2, Pv, Py):
    
    """
    Two lineage transmission model equations. 
    """
    S, Iv, Iy, Rcv, Rcy, Rv, Ry, Jv, Jy, R, _, _ = z
    N = S + Iv + Iy + Rcv + Rcy + Rv + Ry + Jv + Jy + R

    '''We will upload the complete code here once the manuscript is officially published'''
    
    return jnp.stack([dS_dt, dIv_dt, dIy_dt, dRcv_dt, dRcy_dt, dRv_dt, dRy_dt, dJv_dt, dJy_dt, dR_dt, dVnew_dt, dYnew_dt])

# import surveillence data
data = pd.read_csv(r"../data/flu_incidence_us.csv", index_col = 0)
data['Victoria_IR'] = data['ILI%'] * data['positive_rate'] * data['Victoria_proporation'] * 100000
data['Yamagata_IR'] = data['ILI%'] * data['positive_rate'] * data['Yamagata_proporation'] * 100000
data['Victoria_IR_rolling'] = data['Victoria_IR'].rolling(window=4).mean().round()
data['Yamagata_IR_rolling'] = data['Yamagata_IR'].rolling(window=4).mean().round()
data = data.dropna(subset=['Victoria_IR_rolling','Yamagata_IR_rolling'])
data = data.iloc[:153, :].copy()
data.reset_index(inplace=True, drop=True)
data[['Victoria_IR_rolling','Yamagata_IR_rolling']] = data[['Victoria_IR_rolling','Yamagata_IR_rolling']].replace(0, 1e-99)
data

# Define cases for MCMC
y_v = jnp.array(data['Victoria_IR_rolling'].values)
y_y = jnp.array(data['Yamagata_IR_rolling'].values)
y = [y_v, y_y]

def step_time(step):
    
    '''transfer step to time'''

    step_time = 2015 + (43 / 52) + ((step-1) / 52)
    
    return step_time

steps = y_v.shape[0]

def generate_data(S_rate_ini, Iv_rate_ini, Iy_rate_ini, Rcv_rate_ini, Rcy_rate_ini,
                  Rv_rate_ini, Ry_rate_ini, Jv_rate_ini, Jy_rate_ini, R_rate_ini,
                  R0_v, R0_y, w_v, w_y, cv, cy, tv_0, ty_0, immue_esp_pct_v1, immue_esp_pct_v2, immue_esp_pct_v3, 
                  immue_esp_pct_y1, steps, npi_intensity):

    '''We will upload the complete code here once the manuscript is officially published'''
    # integrate dz/dt, z will have shape M x 10
    z = odeint(dz_dt, z_init, ts, w_v, w_y, R0_v, R0_y, D, ksi, Lv, Ly, omega, cv, cy, immue_esp_pct_v1, immue_esp_pct_v2, immue_esp_pct_v3, immue_esp_pct_y1, immue_esp_pct_y2, npi_intensity, npi_start, npi_end, tv_0, ty_0, T, immue_esp_vt1, immue_esp_vt2, immue_esp_vt3, immue_esp_yt1, immue_esp_yt2, Pv, Py, rtol=1e-6, atol=1e-5, mxstep=1000)

    return z

def params_estimate(S_rate_ini, Iv_rate_ini, Iy_rate_ini, Rcv_rate_ini, Rcy_rate_ini,
                    Rv_rate_ini, Ry_rate_ini, Jv_rate_ini, Jy_rate_ini, R_rate_ini,
                    R0_v, R0_y, w_v, w_y, cv, cy, tv_0, ty_0, r, immue_esp_pct_v1, immue_esp_pct_v2, immue_esp_pct_v3, 
                    immue_esp_pct_y1, steps, npi_intensity, file_name):
    
    '''We will upload the complete code here once the manuscript is officially published'''

# import estimated params
df_p = pd.read_csv(r'../data/B_params_rank.csv', index_col=0)
df_r = pd.read_csv(r'../data/params_r0_rank.csv', index_col=0)

list_params = ['w_v', 'w_y', 'cv', 'cy', 'tv_0', 'ty_0', 'r']
list_state = ['S_rate_ini', 'Rcv_rate_ini', 'Rcy_rate_ini', 'Rv_rate_ini', 
             'Ry_rate_ini', 'R_rate_ini', 'Iv_rate_ini', 'Iy_rate_ini', 'Jv_rate_ini', 'Jy_rate_ini']

dict_params = {}

# determine the value of initial compartment BV and BY
for param in list_state:
    param_mean = param + '_adjust_mean'
    dict_params[param] = df_p.loc[0, param_mean]

# determine the value of params BV and BY
for param in list_params:
    param_mean = param + '_mean'
    dict_params[param] = df_p.loc[0, param_mean]

# determine the value of R0
dict_params['R0_v'] = df_r.loc[0, 'R0_v']
dict_params['R0_y'] = df_r.loc[0, 'R0_y']

params_estimate(S_rate_ini = dict_params['S_rate_ini'], Iv_rate_ini = dict_params['Iv_rate_ini'], 
                Iy_rate_ini = dict_params['Iy_rate_ini'], 
                Rcv_rate_ini = dict_params['Rcv_rate_ini'], Rcy_rate_ini = dict_params['Rcy_rate_ini'],
                Rv_rate_ini = dict_params['Rv_rate_ini'], Ry_rate_ini = dict_params['Ry_rate_ini'], 
                Jv_rate_ini = dict_params['Jv_rate_ini'], Jy_rate_ini = dict_params['Jy_rate_ini'], 
                R_rate_ini = dict_params['R_rate_ini'],
                R0_v = dict_params['R0_v'], R0_y = dict_params['R0_y'],
                w_v = float(sys.argv[1]), w_y = float(sys.argv[2]), 
                cv = float(sys.argv[3]), cy = float(sys.argv[4]), tv_0 = float(sys.argv[5]), ty_0 = float(sys.argv[6]), 
                r = float(sys.argv[7]), file_name = sys.argv[8],
                immue_esp_pct_v1 = 0., immue_esp_pct_v2 = 0., immue_esp_pct_v3 = 0., immue_esp_pct_y1 = 0.0,
                npi_intensity = 0., steps=y_v.shape[0]+1)
