# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 18:12:13 2023

@author: Julien Demange-Chryst
"""

#%% Modules

import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append("../cross_entropy/")
from CE_vMFNM import CEIS_vMFNM
from CE_VAE import CEIS_VAE
from iCE_VAE import iCEIS_VAE
from iCE_vMFNM import iCE_vMFNM

ot.Log.Show(ot.Log.NONE)

#%% Test function

input_dim = 50
rho = 1/4

def portfolio(x):
    
    xnp = np.sqrt(9-8*rho**2)*np.array(x)
    d = len(xnp)
    
    threshold = np.sqrt(d)/2
    arr = (xnp>threshold)
    
    return [np.sum(arr)]

ot_portfolio = ot.PythonFunction(input_dim,1,portfolio)

t = input_dim/3

#%% Input distribution

nu = 20
rho = 1/4
sigma2 = (9-8*rho**2)*np.ones(input_dim)
sigma = ot.Point(np.sqrt(sigma2))

R = rho**2/(9-8*rho**2) * np.ones((input_dim,input_dim)) + (1-rho**2/(9-8*rho**2))*np.eye(input_dim)
R = ot.CorrelationMatrix(R)

#input_distr = ot.Student(nu,ot.Point(input_dim),sigma,R)
input_distr = ot.Student(nu,ot.Point(input_dim),ot.Point(np.ones(input_dim)),R)


#%%

X = input_distr.getSample(10**6)
Y = np.array(ot_portfolio(X))
print(np.mean(Y>t))

#%% iCE-VAE

N=10**4

#proba, samples, N_tot = iCEIS_VAE(N, ot_portfolio, t, input_distr, 2, latent_dim=2, K=75)
proba, samples, N_tot = CEIS_VAE(N,.15, ot_portfolio, t, input_distr,latent_dim=2, K=75)
print(f"Estimated failure probability is {proba} with {N_tot} calls to phi.")


#%% Multiple runs


n_rep = 10**2
N_ce = 10**4
N_ice = 5*10**3
delta_cv = 2
p = 0.15

proba_ceis_vae = np.zeros(n_rep)
#proba_iceis_vae = np.zeros(n_rep)
N_tots = np.zeros(n_rep)

#%%

n_start = 0

for n in tqdm(range(n_start,n_rep)):
    proba_vae,_,n_tot = CEIS_VAE(N_ce,p,ot_portfolio,t,input_distr,latent_dim=2,K=75)
    proba_ceis_vae[n] = proba_vae
    N_tots[n] = n_tot
    
   
    # proba_vae,_,n_tot = iCEIS_VAE(N_ice, ot_portfolio, t, input_distr, delta_cv, latent_dim=2, K=75)
    # proba_iceis_vae[n] = proba_vae
    # N_tots[n,1] = n_tot
    
#%%
    
np.savez(f"Data/portfolio_failprob_estimations_{input_dim}.npz",
         CE_vae=proba_ceis_vae,
         N_tots=N_tots)