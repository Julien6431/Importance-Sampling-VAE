# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 17:27:39 2023

@author: Julien Demange-Chryst
"""

#%% Modules

import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append("../cross_entropy/")
from CE_SG import CEIS_SG
from CE_GM import CEIS_GM
from CE_VAE import CEIS_VAE

ot.Log.Show(ot.Log.NONE)

#%% Test function

def four_branchs(x):
    
    xnp = np.array(x)
    d = len(xnp)
    
    side_1 = 1/np.sqrt(d)*np.sum(xnp)
    side_2 = -1/np.sqrt(d)*np.sum(xnp)
    side_3 = 1/np.sqrt(d)*(np.sum(xnp[:d//2]) - np.sum(xnp[d//2:]))
    side_4 = 1/np.sqrt(d)*(-np.sum(xnp[:d//2]) + np.sum(xnp[d//2:]))
    
    return [-np.min([side_1,side_2,side_3,side_4])]

t = 3.5

#%% 2D plot of the function

input_dim = 2
latent_dim = 2
N = 10**6

x1_min = -5
x1_max = 5
x2_min = -5
x2_max = 5

n_points = 10**2+1

x1 = np.linspace(x1_min,x1_max, n_points)
x2 = np.linspace(x2_min,x2_max, n_points)

X1, X2 = np.meshgrid(x1, x2)

values_function = np.zeros((n_points,n_points))
for i in tqdm(range(n_points)):
    for j in range(n_points):
        x = np.array([x1[j],x2[i]])
        values_function[i,j] = four_branchs(x)[0]
        
ot_function = ot.PythonFunction(input_dim,1,four_branchs)
X = ot.Normal(input_dim).getSample(N)
Y = np.array(ot_function(X))

print(f"Failure probability : {np.mean(Y>t)}")

idx = np.where(Y>t)[0]
Xnp = np.array(X)
X_failure = Xnp[idx]

fig,ax = plt.subplots(figsize=(9,9))
cnt = ax.contourf(X1, X2, values_function,levels=100)
ax.contour(X1, X2, values_function, [t],colors='purple')
ax.scatter(X_failure[:,0],X_failure[:,1],color='red',s=6)

#%%

input_dim = 100
ot_function = ot.PythonFunction(input_dim,1,four_branchs)
input_distr = ot.Normal(input_dim)

N=10**4
p=0.25

proba, samples, N_tot = CEIS_VAE(N,p,ot_function,t,input_distr,latent_dim=2,K=75)
#proba, samples,_,N_tot = CEIS_GM(N,p,ot_function,t,input_distr,4)

print(f"Estimated failure probability : {proba}")

#%%

fig, ax = plt.subplots(figsize=(15,15))
ax.contour(X1, X2, values_function, [t], colors="r", linewidths=3)
for sample in samples:
    ax.plot(*np.array(sample).T, ".", markersize=4)
    
#fig.savefig("Figures/four_branches_ce_samples_dim2.png",bbox_inches='tight')


#%%

input_dim = 100
ot_function = ot.PythonFunction(input_dim,1,four_branchs)
input_distr = ot.Normal(input_dim)

n_rep = 10**2
N = 10**4
p = 0.2

proba_ceis_vae = np.zeros(n_rep)
proba_ceis_sg = np.zeros(n_rep)
proba_ceis_gm = np.zeros(n_rep)

for n in tqdm(range(n_rep)):
    proba_vae,_,_ = CEIS_VAE(N,p,ot_function,t,input_distr,latent_dim=2,K=75)
    proba_ceis_vae[n] = proba_vae
    
    proba_sg,_,_,_ = CEIS_SG(N,p,ot_function,t,input_distr)
    proba_ceis_sg[n] = proba_sg
    
    proba_gm,_,_,_ = CEIS_GM(N,p,ot_function,t,input_distr,4)
    proba_ceis_gm[n] = proba_gm
    
    
#np.savez(f"Data/four_branches_failprob_estimations_{input_dim}.npz")