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
numpyro.set_host_device_count(2)


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

def flub_model(M, w_v_mean, w_y_mean, 
               cv_mean, cy_mean, tv_0_mean, ty_0_mean, r_mean,
               S_rate_ini_mean, Rcv_rate_ini_mean, Rcy_rate_ini_mean, 
               Rv_rate_ini_mean, Ry_rate_ini_mean, R_rate_ini_mean, 
               y=[None, None]):
    """
    Main model: which defines priors for the parameters and
    initial populations, integrates the populations over time
    and samples
    M: number of datapoints
    y: datapoints in log scale (S-Iv-Iy-Rcv-Rcy-Rv-Ry-Jv-Jy-R populations in numpy array)
    """    

    # initial state proportions, not treated as parameters
    N=100000.
    
    # measurement times
    ts = jnp.arange(float(M+1))

    '''We will upload the complete code here once the manuscript is officially published'''

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

def param_estimate(w_v_value, w_y_value, 
                   cv_value, cy_value, tv_0_value, ty_0_value, r_value,
                   S_rate_ini_value, Rcv_rate_ini_value, Rcy_rate_ini_value, 
                   Rv_rate_ini_value, Ry_rate_ini_value, R_rate_ini_value,
                   file_name):
    '''estimate the params of ODE systems'''
    '''We will upload the complete code here once the manuscript is officially published'''

param_estimate(  
               w_v_value=float(sys.argv[1]), w_y_value=float(sys.argv[2]), 
               cv_value=float(sys.argv[3]), cy_value=float(sys.argv[4]), 
               tv_0_value=float(sys.argv[5]), ty_0_value=float(sys.argv[6]),
               r_value=float(sys.argv[7]),
               S_rate_ini_value=float(sys.argv[8]), 
               Rcv_rate_ini_value=float(sys.argv[9]), Rcy_rate_ini_value=float(sys.argv[10]), 
               Rv_rate_ini_value=float(sys.argv[11]), Ry_rate_ini_value=float(sys.argv[12]),
               R_rate_ini_value=float(sys.argv[13]),
               file_name=sys.argv[14])
