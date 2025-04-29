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

def dz_dt(z, t, w_v, w_y, R0_v, R0_y, D, ksi, Lv, Ly, omega, cv, cy, immue_esp_pct_v1, immue_esp_pct_v2, immue_esp_pct_v3, immue_esp_pct_y1, immue_esp_pct_y2, npi_intensity, npi_start, npi_end, tv_0, ty_0, T, immue_esp_vt1, immue_esp_vt2, immue_esp_vt3, immue_esp_yt1, immue_esp_yt2, Pv, Py):
    
    """
    Two lineage transmission model equations. 
    """
    S, Iv, Iy, Rcv, Rcy, Rv, Ry, Jv, Jy, R, _, _ = z
    N = S + Iv + Iy + Rcv + Rcy + Rv + Ry + Jv + Jy + R
 
    # set import based on npi_intensity and epidemics
    import_v = 2.
    import_y = 2.

    # seasonal forced transmission rate
    sea_v = 1. + w_v * jnp.cos(2. * jnp.pi * (t - tv_0) / T)
    sea_y = 1. + w_y * jnp.cos(2. * jnp.pi * (t - ty_0) / T)

    # npi intensity
    kappa = 1.

    # infection ability, 不加import?
    lambda_v = kappa * (R0_v / D) * sea_v * (Iv + ksi * Jv) / N
    lambda_y = kappa * (R0_y / D) * sea_y * (Iy + ksi * Jy) / N

    # escape population
    esp_pct_v1 = jnp.where((jnp.abs(t - immue_esp_vt1) <= 1.), immue_esp_pct_v1, 0)
    esp_pct_v2 = jnp.where((jnp.abs(t - immue_esp_vt2) <= 1.), immue_esp_pct_v2, 0)
    esp_pct_v3 = jnp.where((jnp.abs(t - immue_esp_vt3) <= 1.), immue_esp_pct_v3, 0)
    esp_pct_y1 = jnp.where((jnp.abs(t - immue_esp_yt1) <= 1.), immue_esp_pct_y1, 0)
    esp_pct_y2 = jnp.where((jnp.abs(t - immue_esp_yt2) <= 1.), immue_esp_pct_y2, 0)
    
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

    # initial state
    S_rate_ini   = numpyro.sample("S_rate_ini", dist.TruncatedNormal(low=0.1, high=0.9, loc=S_rate_ini_mean, scale=0.02))
    # Iv_rate_ini  = numpyro.sample("Iv_rate_ini", dist.TruncatedNormal(low=0.00001, high=0.00002, loc=0.00001, scale=0.00001))
    # Iy_rate_ini  = numpyro.sample("Iy_rate_ini", dist.TruncatedNormal(low=0.00001, high=0.00002, loc=0.00001, scale=0.00001))
    Rcv_rate_ini = numpyro.sample("Rcv_rate_ini", dist.TruncatedNormal(low=0.01, high=0.1, loc=Rcv_rate_ini_mean, scale=0.02))
    Rcy_rate_ini = numpyro.sample("Rcy_rate_ini", dist.TruncatedNormal(low=0.01, high=0.1, loc=Rcy_rate_ini_mean, scale=0.02))
    Rv_rate_ini  = numpyro.sample("Rv_rate_ini", dist.TruncatedNormal(low=0.1, high=0.9, loc=Rv_rate_ini_mean, scale=0.02))
    Ry_rate_ini  = numpyro.sample("Ry_rate_ini", dist.TruncatedNormal(low=0.1, high=0.9, loc=Ry_rate_ini_mean, scale=0.02))
    # Jv_rate_ini  = numpyro.sample("Jv_rate_ini", dist.TruncatedNormal(low=0.00001, high=0.00002, loc=0.00001, scale=0.00001))
    # Jy_rate_ini  = numpyro.sample("Jy_rate_ini", dist.TruncatedNormal(low=0.00001, high=0.00002, loc=0.00001, scale=0.00001))
    R_rate_ini   = numpyro.sample("R_rate_ini", dist.TruncatedNormal(low=0.1, high=0.9, loc=R_rate_ini_mean, scale=0.02))
    
    Iv_rate_ini  = 1. / N
    Iy_rate_ini  = 1. / N
    Jv_rate_ini  = 1. / N
    Jy_rate_ini  = 1. / N
    # adjust proportion
    per_adjust = 1 / (S_rate_ini + Iv_rate_ini + Iy_rate_ini + Rcv_rate_ini + Rcy_rate_ini + Rv_rate_ini + Ry_rate_ini + 
                      Jv_rate_ini + Jy_rate_ini + R_rate_ini)

    S_ini   = N * S_rate_ini   * per_adjust
    Iv_ini  = N * Iv_rate_ini  * per_adjust
    Iy_ini  = N * Iy_rate_ini  * per_adjust
    Rcv_ini = N * Rcv_rate_ini * per_adjust
    Rcy_ini = N * Rcy_rate_ini * per_adjust
    Rv_ini  = N * Rv_rate_ini  * per_adjust
    Ry_ini  = N * Ry_rate_ini  * per_adjust
    Jv_ini  = N * Jv_rate_ini  * per_adjust
    Jy_ini  = N * Jy_rate_ini  * per_adjust
    R_ini   = N * R_rate_ini   * per_adjust
    
    z_init  = jnp.array([S_ini, Iv_ini, Iy_ini, Rcv_ini, Rcy_ini, Rv_ini, Ry_ini, Jv_ini, Jy_ini, R_ini, 0.0, 0.0])

    # unfixed parameters
    R0_v  = 1.3
    R0_y  = 1.2
    # R0_v  = numpyro.sample("R0_v", dist.TruncatedNormal(low=1.0, high=3.0, loc=R0_v_mean, scale=0.02))
    # R0_y  = numpyro.sample("R0_y", dist.TruncatedNormal(low=1.0, high=3.0, loc=R0_y_mean, scale=0.02))
    w_v   = numpyro.sample("w_v", dist.TruncatedNormal(low=0.01, high=1.0, loc=w_v_mean, scale=0.02))
    w_y   = numpyro.sample("w_y", dist.TruncatedNormal(low=0.01, high=1.0, loc=w_y_mean, scale=0.02))
    cv    = numpyro.sample("cv", dist.TruncatedNormal(low=0.01, high=1.0, loc=cv_mean, scale=0.02))
    cy    = numpyro.sample("cy", dist.TruncatedNormal(low=0.01, high=1.0, loc=cy_mean, scale=0.02))
    tv_0    = numpyro.sample("tv_0", dist.TruncatedNormal(low=12., high=22., loc=tv_0_mean, scale=0.2))
    ty_0    = numpyro.sample("ty_0", dist.TruncatedNormal(low=12., high=22., loc=ty_0_mean, scale=0.2))
    r = numpyro.sample("r", dist.TruncatedNormal(low=0.01, high=1., loc=r_mean, scale=0.02))
    immue_esp_pct_v1 = 0.
    immue_esp_pct_v2 = 0.
    
    # fixed parameters
    mu = float(0.0098/52)
    npi_start = 218.
    npi_end = 285.
    npi_intensity = 1.
    T = float(52.)
    immue_esp_vt1 = float(101)
    immue_esp_vt2 = float(205)
    immue_esp_vt3 = float(310)
    immue_esp_yt1 = float(205)
    immue_esp_yt2 = float(310) 
    immue_esp_pct_v3 = 0.2
    immue_esp_pct_y1 = 0.
    immue_esp_pct_y2 = 0.

    D  = 3.4 / 7.
    ksi = 1.
    Lv = 4. * 52
    Ly = 4. * 52
    Pv = (1./12) * 52
    Py = (1./12) * 52
    omega = 1.

    # integrate dz/dt, z will have shape M x 10
    z = odeint(dz_dt, z_init, ts, w_v, w_y, R0_v, R0_y, D, ksi, Lv, Ly, omega, cv, cy, immue_esp_pct_v1, immue_esp_pct_v2, immue_esp_pct_v3, immue_esp_pct_y1, immue_esp_pct_y2, npi_intensity, npi_start, npi_end, tv_0, ty_0, T, immue_esp_vt1, immue_esp_vt2, immue_esp_vt3, immue_esp_yt1, immue_esp_yt2, Pv, Py, rtol=1e-6, atol=1e-5, mxstep=1000)
    
    # S, Iv, Iy, Rcv, Rcy, Rv, Ry, Jv, Jy, R, _, _
    S = numpyro.deterministic('S', z[:,0])
    Iv = numpyro.deterministic('Iv', z[:,1])
    Iy = numpyro.deterministic('Iy', z[:,2])
    Rcv = numpyro.deterministic('Rcv', z[:,3])
    Rcy = numpyro.deterministic('Rcy', z[:,4])
    Rv = numpyro.deterministic('Rv', z[:,5])
    Ry = numpyro.deterministic('Ry', z[:,6])
    Jv = numpyro.deterministic('Jv', z[:,7])
    Jy = numpyro.deterministic('Jy', z[:,8])
    R = numpyro.deterministic('R', z[:,9])
    C_BV = numpyro.deterministic('C_BV', z[:,10])
    C_BY = numpyro.deterministic('C_BY', z[:,11])
    
    # new cases
    inf_hat_v = jnp.diff(z[:,10])
    inf_hat_y = jnp.diff(z[:,11])
    
    # measured populations (in log scale)
    numpyro.sample("inf_hat_v", dist.Poisson(r * inf_hat_v), obs=y[0])
    numpyro.sample("inf_hat_y", dist.Poisson(r * inf_hat_y), obs=y[1])

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
    # start time
    start_time = time.time()
    
    # params estimate
    kernel = NUTS(flub_model, dense_mass=True, max_tree_depth=10, target_accept_prob=0.80)
    mcmc = MCMC(
    sampler=kernel, 
    num_warmup=500, 
    num_samples=500, 
    num_chains=2,
    progress_bar=True,
    )
    key = jax.random.PRNGKey(0)
    mcmc.run(key, M=y_v.shape[0], 
             w_v_mean=w_v_value, w_y_mean=w_y_value, 
             cv_mean=cv_value, cy_mean=cy_value,
             tv_0_mean=tv_0_value, ty_0_mean=ty_0_value, 
             r_mean=r_value,
             S_rate_ini_mean=S_rate_ini_value, 
             Rcv_rate_ini_mean=Rcv_rate_ini_value, Rcy_rate_ini_mean=Rcy_rate_ini_value, 
             Rv_rate_ini_mean=Rv_rate_ini_value, Ry_rate_ini_mean=Ry_rate_ini_value, 
             R_rate_ini_mean=R_rate_ini_value, 
             y=[y_v, y_y])
    # params
    params_orign = ['w_v', 'w_y', 'cv', 'cy', 
                   'tv_0', 'ty_0', 'r',
                   'S_rate_ini', 'Rcv_rate_ini', 'Rcy_rate_ini', 
                   'Rv_rate_ini', 'Ry_rate_ini', 'R_rate_ini']
    
    mcmc_samples = mcmc.get_samples()
    az_trace = az.from_numpyro(mcmc)
    az_summary = az.summary(az_trace, var_names=params_orign)

    # estimated original params
    params_dict = {}
    for param in params_orign:
        param_mean_value = jnp.mean(mcmc_samples[param])
        param_std_value = jnp.std(mcmc_samples[param])
        params_dict[param + '_mean'] = param_mean_value
        params_dict[param + '_std'] = param_std_value

    # save params to a dataframe
    df_params_orign = pd.DataFrame(params_dict, index=[0])
    df_params_orign.to_csv(f'./result/dynamic_model/params/orign/params_group{file_name}.csv', index=False)

    # estimated params by az
    # params dict
    data_params = {f"{var}_mean": [az_summary.loc[var, 'mean']] for var in params_orign}
    data_params.update({f"{var}_std": [az_summary.loc[var, 'sd']] for var in params_orign})
    data_params.update({f"{var}_neff": [az_summary.loc[var, 'ess_bulk']] for var in params_orign})
    data_params.update({f"{var}_rhat": [az_summary.loc[var, 'r_hat']] for var in params_orign})

    # likelihood
    log_likelihood_bv = az_trace.log_likelihood.inf_hat_v.values.sum(axis=2)
    data_params['bv_log_likelihood_mean'] = [log_likelihood_bv.mean()]
    data_params['bv_log_likelihood_std'] = [log_likelihood_bv.std()]

    log_likelihood_by = az_trace.log_likelihood.inf_hat_y.values.sum(axis=2)
    data_params['by_log_likelihood_mean'] = [log_likelihood_by.mean()]
    data_params['by_log_likelihood_std'] = [log_likelihood_by.std()]

    # end time
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 3600
    data_params['runtime'] = [elapsed_time]

    # save params to a dataframe
    df_params = pd.DataFrame(data_params)

    # posterior dict
    posterior_params = {f"{var}_chain1": list(az_trace['posterior'][var].values[0]) for var in params_orign}
    posterior_params.update({f"{var}_chain2": list(az_trace['posterior'][var].values[1]) for var in params_orign})

    # save params to a dataframe
    df_posterior = pd.DataFrame(posterior_params)

    # export data
    df_posterior.to_csv(f'./result/dynamic_model/posterior/params_group{file_name}.csv', index=False)

    # plot trace
    az.plot_trace(az_trace, var_names=params_orign)
    plt.tight_layout()
    plt.savefig(f'./result/dynamic_model/figure/trace/params_group{file_name}.png')

    # plot posterior
    az.plot_posterior(az_trace, var_names=params_orign, grid=(5,5))
    plt.tight_layout()
    plt.savefig(f'./result/dynamic_model/figure/posterior/params_group{file_name}.png')
    

    # fitting plot
    ss = Predictive(flub_model, mcmc_samples)(key, M=y_v.shape[0], 
             w_v_mean=w_v_value, w_y_mean=w_y_value, 
             cv_mean=cv_value, cy_mean=cy_value,
             tv_0_mean=tv_0_value, ty_0_mean=ty_0_value, 
             r_mean=r_value,
             S_rate_ini_mean=S_rate_ini_value, 
             Rcv_rate_ini_mean=Rcv_rate_ini_value, Rcy_rate_ini_mean=Rcy_rate_ini_value, 
             Rv_rate_ini_mean=Rv_rate_ini_value, Ry_rate_ini_mean=Ry_rate_ini_value, 
             R_rate_ini_mean=R_rate_ini_value)

    inf_hat_v = ss['inf_hat_v']
    inf_hat_y = ss['inf_hat_y']
    S = ss['S']
    Iv = ss['Iv']
    Iy = ss['Iy']
    Rcv = ss['Rcv']
    Rcy = ss['Rcy']
    Rv = ss['Rv']
    Ry = ss['Ry']
    Jv = ss['Jv']
    Jy = ss['Jy']
    R = ss['R']
    C_BV = ss['C_BV']
    C_BY = ss['C_BY']

    # mean value
    mu_bv = jnp.median(inf_hat_v, 0)
    pi_bv = jnp.percentile(inf_hat_v, jnp.array([5, 95]), 0)
    
    mu_by = jnp.median(inf_hat_y, 0)
    pi_by = jnp.percentile(inf_hat_y, jnp.array([5, 95]), 0)
    
    mu_S = jnp.median(S, 0)
    pi_S = jnp.percentile(S, jnp.array([5, 95]), 0)
    
    mu_Iv = jnp.median(Iv, 0)
    pi_Iv = jnp.percentile(Iv, jnp.array([5, 95]), 0)
    
    mu_Iy = jnp.median(Iy, 0)
    pi_Iy = jnp.percentile(Iy, jnp.array([5, 95]), 0)
    
    mu_Rcv = jnp.median(Rcv, 0)
    pi_Rcv = jnp.percentile(Rcv, jnp.array([5, 95]), 0)
    
    mu_Rcy = jnp.median(Rcy, 0)
    pi_Rcy = jnp.percentile(Rcy, jnp.array([5, 95]), 0)
    
    mu_Rv = jnp.median(Rv, 0)
    pi_Rv = jnp.percentile(Rv, jnp.array([5, 95]), 0)
    
    mu_Ry = jnp.median(Ry, 0)
    pi_Ry = jnp.percentile(Ry, jnp.array([5, 95]), 0)
    
    mu_Jv = jnp.median(Jv, 0)
    pi_Jv = jnp.percentile(Jv, jnp.array([5, 95]), 0)
    
    mu_Jy = jnp.median(Jy, 0)
    pi_Jy = jnp.percentile(Jy, jnp.array([5, 95]), 0)
    
    mu_R = jnp.median(R, 0)
    pi_R = jnp.percentile(R, jnp.array([5, 95]), 0)
    
    mu_C_BV = jnp.median(C_BV, 0)
    pi_C_BV = jnp.percentile(C_BV, jnp.array([5, 95]), 0)
    
    mu_C_BY = jnp.median(C_BY, 0)
    pi_C_BY = jnp.percentile(C_BY, jnp.array([5, 95]), 0)

    # fitting metrics
    bv_rmse = root_mean_squared_error(y_v, mu_bv)
    by_rmse = root_mean_squared_error(y_y, mu_by)
    
    bv_mape = mean_absolute_percentage_error(y_v, mu_bv)
    by_mape = mean_absolute_percentage_error(y_y, mu_by)
    
    bv_mae = mean_absolute_error(y_v, mu_bv)
    by_mae = mean_absolute_error(y_y, mu_by)
    
    bv_corr = jnp.corrcoef(y_v, mu_bv)[0,1]
    by_corr = jnp.corrcoef(y_y, mu_by)[0,1]
    
    bv_r2 = r2_score(np.array(y_v), np.array(mu_bv))
    by_r2 = r2_score(np.array(y_y), np.array(mu_by))
    
    df_params['bv_rmse'] = bv_rmse
    df_params['by_rmse'] = by_rmse

    df_params['bv_mape'] = bv_mape
    df_params['by_mape'] = by_mape

    df_params['bv_mae'] = bv_mae
    df_params['by_mae'] = by_mae

    df_params['bv_corr'] = bv_corr
    df_params['by_corr'] = by_corr

    df_params['bv_r2'] = bv_r2
    df_params['by_r2'] = by_r2
    
    # export data
    df_params.to_csv(f'./result/dynamic_model/params/az/params_az_group{file_name}.csv', index=False)

    # ts = jnp.arange(float(y_v.shape[0]+1000))
    ts = jnp.array([step_time(i) for i in range(1, y_v.shape[0]+1)], dtype='float64')
    ts_part = jnp.array([step_time(i) for i in range(1, y_v.shape[0]+2)], dtype='float64')
    ts_true = jnp.array([step_time(i) for i in range(1, y_v.shape[0]+1)], dtype='float64')
    
    plt.figure(figsize=(21, 12), constrained_layout=True)
    plt.subplot(331)
    plt.plot(ts, mu_bv, ls="--", color='#a9373b', label="predict new_bv", lw=2, alpha=1)
    plt.plot(ts_true, y_v, ls="-", color='#a9373b', label="true new_bv", lw=2, alpha=1)
    plt.legend(frameon=False)
    plt.title("BV new cases")
    # plt.fill_between(ts, pi_bv[0], pi_bv[1], color="C1", alpha=0.2)
    
    plt.subplot(334)
    plt.plot(ts, mu_by, ls="--", color='#2369bd', label="predict new_by", lw=2, alpha=1)
    plt.plot(ts_true, y_y, ls="-", color='#2369bd', label="true new_by", lw=2, alpha=1)
    plt.legend(frameon=False)
    plt.title("BY new cases")
    
    plt.subplot(332)
    plt.plot(ts_part, mu_S, ls="-",  label="S", lw=2, alpha=1)
    plt.plot(ts_part, mu_Iv, ls="-",  label="Iv", lw=2, alpha=1)
    plt.plot(ts_part, mu_Iy, ls="-",  label="Iy", lw=2, alpha=1)
    plt.plot(ts_part, mu_Rcv, ls="-",  label="Rcv", lw=2, alpha=1)
    plt.plot(ts_part, mu_Rcy, ls="-",  label="Rcy", lw=2, alpha=1)
    plt.plot(ts_part, mu_Rv, ls="-",  label="Rv", lw=2, alpha=1)
    plt.plot(ts_part, mu_Ry, ls="-",  label="Ry", lw=2, alpha=1)
    plt.plot(ts_part, mu_Jv, ls="-",  label="Jv", lw=2, alpha=1)
    plt.plot(ts_part, mu_Jy, ls="-",  label="Jy", lw=2, alpha=1)
    plt.plot(ts_part, mu_R, ls="-",  label="R", lw=2, alpha=1)
    plt.legend(frameon=False)
    plt.title("compartment")
    
    plt.subplot(333)
    plt.plot(ts_part, mu_S, ls="-",  label="S", lw=2, alpha=1)
    # plt.plot(ts_part, mu_Iv, ls="-",  label="Iv", lw=2, alpha=1)
    # plt.plot(ts_part, mu_Iy, ls="-",  label="Iy", lw=2, alpha=1)
    plt.plot(ts_part, mu_Rcv, ls="-",  label="Rcv", lw=2, alpha=1)
    plt.plot(ts_part, mu_Rcy, ls="-",  label="Rcy", lw=2, alpha=1)
    plt.plot(ts_part, mu_Rv, ls="-",  label="Rv", lw=2, alpha=1)
    plt.plot(ts_part, mu_Ry, ls="-",  label="Ry", lw=2, alpha=1)
    # plt.plot(ts_part, mu_Jv, ls="-",  label="Jv", lw=2, alpha=1)
    # plt.plot(ts_part, mu_Jy, ls="-",  label="Jy", lw=2, alpha=1)
    plt.plot(ts_part, mu_R, ls="-",  label="R", lw=2, alpha=1)
    plt.legend(frameon=False)
    plt.title("compartment")
    
    plt.subplot(335)
    # plt.plot(ts_part, mu_S, ls="-",  label="S", lw=2, alpha=1)
    plt.plot(ts_part, mu_Iv, ls="-",  label="Iv", lw=2, alpha=1)
    plt.plot(ts_part, mu_Iy, ls="-",  label="Iy", lw=2, alpha=1)
    # plt.plot(ts_part, mu_Rcv, ls="-",  label="Rcv", lw=2, alpha=1)
    # plt.plot(ts_part, mu_Rcy, ls="-",  label="Rcy", lw=2, alpha=1)
    # plt.plot(ts_part, mu_Rv, ls="-",  label="Rv", lw=2, alpha=1)
    # plt.plot(ts_part, mu_Ry, ls="-",  label="Ry", lw=2, alpha=1)
    plt.plot(ts_part, mu_Jv, ls="-",  label="Jv", lw=2, alpha=1)
    plt.plot(ts_part, mu_Jy, ls="-",  label="Jy", lw=2, alpha=1)
    # plt.plot(ts_part, mu_R, ls="-",  label="R", lw=2, alpha=1)
    plt.legend(frameon=False)
    plt.title("compartment")

    plt.subplot(336)
    # plt.plot(ts_part, mu_S, ls="-",  label="S", lw=2, alpha=1)
    # plt.plot(ts_part, mu_Iv, ls="-",  label="Iv", lw=2, alpha=1)
    # plt.plot(ts_part, mu_Iy, ls="-",  label="Iy", lw=2, alpha=1)
    plt.plot(ts_part, mu_Rcv, ls="-",  label="Rcv", lw=2, alpha=1)
    plt.plot(ts_part, mu_Rcy, ls="-",  label="Rcy", lw=2, alpha=1)
    plt.plot(ts_part, mu_Rv, ls="-",  label="Rv", lw=2, alpha=1)
    plt.plot(ts_part, mu_Ry, ls="-",  label="Ry", lw=2, alpha=1)
    # plt.plot(ts_part, mu_Jv, ls="-",  label="Jv", lw=2, alpha=1)
    # plt.plot(ts_part, mu_Jy, ls="-",  label="Jy", lw=2, alpha=1)
    plt.plot(ts_part, mu_R, ls="-",  label="R", lw=2, alpha=1)
    plt.legend(frameon=False)
    plt.title("compartment")
    
    plt.subplot(337)
    plt.plot(ts_part, mu_C_BV, ls="-", color='#a9373b', label="C_BV", lw=2, alpha=1)
    plt.plot(ts_part, mu_C_BY, ls="-", color='#2369bd', label="C_BY", lw=2, alpha=1)
    plt.legend(frameon=False)
    plt.title("commulative cases")
    
    plt.subplot(338)
    plt.plot(ts, jnp.diff(mu_C_BV), ls="--", color='#a9373b', label="predict_BV", lw=2, alpha=1)
    plt.plot(ts_true, y_v, ls="-", color='#a9373b', label="true_BY", lw=2, alpha=1)
    plt.legend(frameon=False)
    plt.title("commulative cases")

    plt.subplot(339)
    plt.plot(ts, jnp.diff(mu_C_BY), ls="--", color='#2369bd', label="predict_BV", lw=2, alpha=1)
    plt.plot(ts_true, y_y, ls="-", color='#2369bd', label="true_BY", lw=2, alpha=1)
    plt.legend(frameon=False)
    plt.title("commulative cases")

    plt.savefig(f'./result/dynamic_model/figure/fitting/params_group{file_name}.png')

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