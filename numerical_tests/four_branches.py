#%% Modules

import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append("../cross_entropy/")
from CE_vMFNM import CEIS_vMFNM
from CE_VAE import CEIS_VAE

ot.Log.Show(ot.Log.NONE)

#%% Four branches test function

def four_branches(x):
    
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
N = 10**4

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
        values_function[i,j] = four_branches(x)[0]
        
ot_function = ot.PythonFunction(input_dim,1,four_branches)
X = ot.Normal(input_dim).getSample(N)

fig,ax = plt.subplots(figsize=(11,9))
cnt = ax.contourf(X1, X2, values_function,levels=100)
ct = ax.contour(X1, X2, values_function, [t],colors='red',linewidths=3)

cbar = fig.colorbar(cnt, ax=ax, ticks=[0,1,2,3,4,5,6,7])
for ta in cbar.ax.get_yticklabels():
     ta.set_fontsize(20)

plt.clabel(ct, inline=1, fontsize=30)
ax.set_xlabel("$X_1$",fontsize=20)
ax.set_ylabel("$X_2$",fontsize=20)
ax.tick_params(axis='x',labelsize='xx-large')
ax.tick_params(axis='y',labelsize='xx-large')

fig.savefig("Figures/four_branches_2d.png",bbox_inches='tight',dpi=500)

#%% CE-VAE

input_dim = 100
ot_function = ot.PythonFunction(input_dim,1,four_branches)
input_distr = ot.Normal(input_dim)

N=10**4
proba, samples, N_tot = CEIS_VAE(N,.25, ot_function, t, input_distr,latent_dim=2, K=75, lat_space_plot=True)
print(f"Estimated failure probability is {proba} with {N_tot} calls to phi.")

if input_dim==2:
    fig, ax = plt.subplots(figsize=(15,15))
    ax.contour(X1, X2, values_function, [t], colors="r", linewidths=3)
    for sample in samples:
        ax.plot(*np.array(samples).T, ".", markersize=10)

    fig.savefig("Figures/four_branches_ce_samples_dim2.png",bbox_inches='tight')


#%% Multiple runs

input_dim = 100
ot_function = ot.PythonFunction(input_dim,1,four_branches)
input_distr = ot.Normal(input_dim)

n_rep = 10**2
N_ce = 10**4
p = 0.25

proba_ceis_vae = np.zeros(n_rep)
proba_ceis_vM = np.zeros(n_rep)
proba_ceis_vM_3 = np.zeros(n_rep)
proba_ceis_vM_5 = np.zeros(n_rep)
N_tots = np.zeros((n_rep,4))

for n in tqdm(range(n_rep)):
    proba_vae,_,n_tot = CEIS_VAE(N_ce,p,ot_function,t,input_distr,latent_dim=2,K=75)
    proba_ceis_vae[n] = proba_vae
    N_tots[n,0] = n_tot
    
    proba_vM,_,n_tot,_,_ = CEIS_vMFNM(N_ce,p,ot_function,t,input_distr,4)
    proba_ceis_vM[n] = proba_vM
    N_tots[n,1] = n_tot
    
    proba_vM,_,n_tot,_,_ = CEIS_vMFNM(N_ce,p,ot_function,t,input_distr,3)
    proba_ceis_vM_3[n] = proba_vM
    N_tots[n,2] = n_tot
    
    proba_vM,_,n_tot,_,_ = CEIS_vMFNM(N_ce,p,ot_function,t,input_distr,5)
    proba_ceis_vM_5[n] = proba_vM
    N_tots[n,3] = n_tot
    
    

#%% Save data
    
np.savez(f"Data/four_branches_failprob_estimations_{input_dim}.npz",
         CE_vae=proba_ceis_vae,
         CE_vM=proba_ceis_vM,
         CE_vae_3=proba_ceis_vM_3,
         CE_vae_5=proba_ceis_vM_5,
         N_tots=N_tots)
