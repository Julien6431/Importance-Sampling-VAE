# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 17:17:27 2023

@author: jdemange
"""

#%% Modules

import numpy as np
import matplotlib.pyplot as plt
import openturns as ot
import pickle

#%% Plot dim 10

file = open('Data/generation_vae_10.pkl', 'rb')
samples_vae = pickle.load(file)
file.close()

file = open('Data/generation_mamisGM_10.pkl', 'rb')
samples_mamisGM = pickle.load(file)
file.close()

#%%

def save_fig_10(algo='vae',nb_mode=2):
    dim = 10
    
    target_mean1 = ot.Point(2.5*np.ones(dim))
    target_mean2 = ot.Point(-2.5*np.ones(dim))
    target_cov_matrix = ot.CovarianceMatrix(np.eye(dim))  
    distrs = [ot.Normal(target_mean1,target_cov_matrix),ot.Normal(target_mean2,target_cov_matrix)]
    target_distr = ot.Mixture(distrs)
    
    init_distr = ot.Normal(dim)
    
    
    xx = np.linspace(-6,6,1001).reshape((-1,1))
    yy = np.array(target_distr.getMarginal(0).computePDF(xx)).flatten()
    yy_init = np.array(init_distr.getMarginal(0).computePDF(xx)).flatten()
    
    if (algo=="vae" and nb_mode==2):
        sample = samples_vae[-2]
        title = "MAMIS-VAE with 2 modes"
    elif (algo=="vae" and nb_mode==1):
        sample = samples_vae[1]
        title = "MAMIS-VAE with 1 mode"
    elif (algo=="mamisGM" and nb_mode==2):
        sample = samples_mamisGM[1]
        title = "MAMIS-GM with 2 modes"
    elif (algo=="mamisGM" and nb_mode==1):
        sample = samples_mamisGM[3]
        title = "MAMIS-GM with 1 mode"
        
        
    fig,ax = plt.subplots(2,5,figsize=(12,5))
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.35,
                    hspace=0.35)
    for i in range(2):
        for j in range(5):
            ax[i,j].hist(np.array(sample)[:,5*i+j],bins=100,density=True)
            ax[i,j].plot(xx,yy)
            ax[i,j].plot(xx,yy_init)
            
            ax[i,j].set_xlabel(r"$X_{{{}}}$".format(5*i+j+1),fontdict={'fontsize': 15,'fontweight' : 'bold'})
                      
    #fig.suptitle(title,fontsize= 20,fontweight='bold')    
    fig.savefig("Figures/plot_generation_"+algo+"_"+str(nb_mode)+".png",bbox_inches='tight',dpi=300)
    
#%%

save_fig_10(algo="vae",nb_mode=2)
save_fig_10(algo="vae",nb_mode=1)
save_fig_10(algo="mamisGM",nb_mode=2)
save_fig_10(algo="mamisGM",nb_mode=1)


#%% Plot dim 20

file = open('Data/generation_vae_20.pkl', 'rb')
samples_vae_20 = pickle.load(file)
file.close()

file = open('Data/generation_vae_20_weights.pkl', 'rb')
samples_weights_20 = pickle.load(file)
file.close()


def save_fig_20(idx):
    dim = 20
    
    distr_1 = ot.Student(4,-2,1)
    distr_2 = ot.LogNormal(0,1)
    distr_3 = ot.Triangular(1,3,5)
        
    left_distrs = [ot.Normal(2,1) for _ in range(dim-3)]
    
    R = ot.CorrelationMatrix(dim)
    for i in range(dim-1):
        R[i, i+1] = 0.25
    copula = ot.NormalCopula(R)
    
    target_distr = ot.ComposedDistribution([distr_1,distr_2,distr_3] + left_distrs,copula)
    
    init_mean = ot.Point(np.zeros(dim))
    init_cov_matrix = ot.CovarianceMatrix(2*np.eye(dim)) 
    init_distr = ot.Normal(init_mean,init_cov_matrix)
    
    
    xx = np.linspace(-6,6,1001).reshape((-1,1))
    yy_init = np.array(init_distr.getMarginal(0).computePDF(xx)).flatten()

    fig,ax = plt.subplots(4,5,figsize=(16,10))
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.35,
                    hspace=0.35)
    
    for i in range(4):
        for j in range(5):
            ax[i,j].hist(np.array(samples_vae_20[idx])[:,5*i+j],bins=100,density=True,weights=samples_weights_20[idx])
            yy = np.array(target_distr.getMarginal(5*i+j).computePDF(xx)).flatten()
            ax[i,j].plot(xx,yy) 
            ax[i,j].plot(xx,yy_init) 
            ax[i,j].set_xlabel(r"$X_{{{}}}$".format(5*i+j+1),fontdict={'fontsize': 15,'fontweight' : 'bold'})
            
        
    fig.savefig("Figures/plot_generation_20_"+str(idx)+".png",bbox_inches='tight',dpi=300) 
    
save_fig_20(0)
save_fig_20(1)
save_fig_20(2)
save_fig_20(3)
    
    
    
    
    
    
    
    