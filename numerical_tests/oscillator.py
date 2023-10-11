"""
@author: Julien Demange-Chryst
"""


#%% Modules

import numpy as np
import openturns as ot
from tqdm import tqdm
import sys
sys.path.append("../cross_entropy/")
from CE_vMFNM import CEIS_vMFNM
from CE_VAE import CEIS_VAE

#%% Oscillator test function

input_dim = 200
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

#when d>=100, the failure probability phi > 0 is 4.28e-04


#%% CE-VAE

input_dim = 200
input_distr = ot.Normal(input_dim)

N=10**4

proba, samples, N_tot = CEIS_VAE(N,.15, phi, t, input_distr,latent_dim=2, K=75)
print(f"Estimated failure probability is {proba} with {N_tot} calls to phi.")
    
    
#%% Multiple runs


input_dim = 200
input_distr = ot.Normal(input_dim)

n_rep = 10**2
N_ce = 10**4
p = 0.15

proba_ceis_vae = np.zeros(n_rep)
proba_ceis_vM_2 = np.zeros(n_rep)
proba_ceis_vM_1 = np.zeros(n_rep)
proba_ceis_vM_3 = np.zeros(n_rep)
N_tots = np.zeros((n_rep,4))

for n in tqdm(range(n_rep)):
    proba_vae,_,n_tot = CEIS_VAE(N_ce,p,phi,t,input_distr,latent_dim=2,K=75)
    proba_ceis_vae[n] = proba_vae
    N_tots[n,0] = n_tot
    
    proba_vM,_,n_tot,_,_ = CEIS_vMFNM(N_ce,p,phi,t,input_distr,2)
    proba_ceis_vM_2[n] = proba_vM
    N_tots[n,1] = n_tot
    
    proba_vM,_,n_tot,_,_ = CEIS_vMFNM(N_ce,p,phi,t,input_distr,1)
    proba_ceis_vM_1[n] = proba_vM
    N_tots[n,2] = n_tot
    
    proba_vM,_,n_tot,_,_ = CEIS_vMFNM(N_ce,p,phi,t,input_distr,3)
    proba_ceis_vM_3[n] = proba_vM
    N_tots[n,3] = n_tot


#%% Save data

np.savez(f"Data/oscillator_failprob_estimations_{input_dim}.npz",
         CE_vae=proba_ceis_vae,
         CE_vM2=proba_ceis_vM_2,
         CE_vM1=proba_ceis_vM_1,
         CE_vM3=proba_ceis_vM_3,
         N_tots=N_tots)