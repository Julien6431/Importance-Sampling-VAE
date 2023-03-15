# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 09:54:47 2023

@author: Julien Demange-Chryst
"""


#%% Modules

import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append("../cross_entropy/")
from CE_VAE import CEIS_VAE

#%% Settings

input_dim = 100
input_distr = ot.Normal(input_dim)

z_0=0
x_0=1.5
m=10**3
c=200*np.pi
k=1000*((2*np.pi)**2)
gamma=1
del_w=30*np.pi/input_dim
S=0.005
sigma=np.sqrt(2*S*del_w)
t_max=2
dt=0.004  
t_k=np.arange(0, t_max, dt)

def f_t2(t,U):
    d_2=int(input_dim/2)
    N=np.shape(U)[0]
    res=np.zeros(N)

    for i in range(d_2):
        w_i=i*del_w
        res=res+U[:,i]*np.cos(w_i*t)+U[:,i+d_2]*np.sin(w_i*t)
        
    return -m*sigma*res

def F_1_2(i,Z,X):
    return X[:,i]

def F_2_2(i,Z,X,U):
    return (-c*X[:,i]-k*(Z[:,i]+gamma*(np.power(Z[:,i],3)))+f_t2(t_k[i],U))/m

def Euler2(U):
    N=np.shape(U)[0]
    len_t=len(t_k)
    Z=np.zeros((N,len_t))
    X=np.zeros((N,len_t))
    Z[:,0]=z_0*np.ones(N)
    X[:,0]=x_0*np.ones(N)
    
    for i in range(len_t-1):
        Z[:,i+1]=Z[:,i]+dt*F_1_2(i,Z,X)
        X[:,i+1]=X[:,i]+dt*F_2_2(i,Z,X,U)
    
    return Z[:,len_t-1]     

def phi(U):
    res=Euler2(U)
    N=np.shape(U)[0]
    matrice=np.zeros((N,2))
    matrice[:,0]=0.1*np.ones(N)-res
    matrice[:,1]=res+0.06*np.ones(N)
    return -np.min(matrice,axis=1)

t = 0

#lorsque d>=100, la proba phi > 0 est à 4.28e-04


#%%

input_dim = 100
input_distr = ot.Normal(input_dim)

N=10**4
p=0.25

proba, samples, N_tot = CEIS_VAE(N,p,phi,t,input_distr,latent_dim=2,K=75)

print(f"Estimated failure probability : {proba}")




