# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:38:28 2023

@author: jdemange
"""

#%% Modules

import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as kb
import time

import sys
sys.path.append("../src/")
from VAE_IS_VP import fitted_vae


#%%


def compute_Dkl(target_distr,simu_distr,mean_x,std_x):
    
    sample = target_distr.getSample(5*10**4)
    log_target = np.array(target_distr.computeLogPDF(sample)).flatten()
    
    sample_np = np.array(sample)
    sample_std = ot.Sample((sample_np - mean_x)/std_x)
        
    log_simu = np.array(simu_distr.computeLogPDF(sample_std)).flatten() - np.sum(np.log(std_x))
    
    return np.mean(log_target - log_simu)
    

def adaptive_is(target_distr,init_distr,N,max_iter,latent_dim=2):
        
    X = init_distr.getSample(N)
    log_targetX = target_distr.computeLogPDF(X)
    log_initX = init_distr.computeLogPDF(X)
    
    W = np.exp(log_targetX - log_initX)
    samples = [X]
    log_W = [log_targetX - log_initX]
    
    for n in range(max_iter):
        #print(n)
        kb.clear_session()
        vae,_,_ = fitted_vae(np.array(X).astype('float32'), W.astype('float32'), latent_dim=latent_dim, K=75)
        
        X,log_gX = vae.getSample(N,with_pdf=True)
        log_targetX = target_distr.computeLogPDF(X)
        W = np.exp(log_targetX-log_gX)
        
        samples.append(X)
        log_W.append(log_targetX-log_gX)
    
        mean_x = vae.mean_x.astype('float64')
        std_x = vae.std_x.astype('float64')
        distrX = vae.distrX
     
         
    return samples,[distrX,mean_x,std_x]

    
#%% Single test


def single_test(dim):
    if dim == 10:
        target_mean1 = ot.Point(2.5*np.ones(dim))
        target_mean2 = ot.Point(-2.5*np.ones(dim))
        
        target_cov_matrix = ot.CovarianceMatrix(np.eye(dim)) 
        
        distrs = [ot.Normal(target_mean1,target_cov_matrix),ot.Normal(target_mean2,target_cov_matrix)]

        target_distr = ot.Mixture(distrs)
        init_mean = ot.Point(np.zeros(dim))
        init_cov_matrix = ot.CovarianceMatrix(1*np.eye(dim)) 
        init_distr = ot.Normal(init_mean,init_cov_matrix)
        samples,vae = adaptive_is(target_distr, init_distr, 10**4, 10,latent_dim=4)


        xx = np.linspace(-6,6,1001).reshape((-1,1))
        yy = np.array(target_distr.getMarginal(0).computePDF(xx)).flatten()
        yy2 = np.array(init_distr.getMarginal(0).computePDF(xx)).flatten()
        
        last_sample = np.array(samples[-1])
        fig,ax = plt.subplots(2,5,figsize=(12,6))
        for i in range(2):
            for j in range(5):
                ax[i,j].hist(last_sample[:,5*i+j],bins=100,density=True)
                ax[i,j].plot(xx,yy)
                ax[i,j].plot(xx,yy2)
                
        print(compute_Dkl(target_distr, vae[0],vae[1],vae[2]))
                
    elif dim == 20:
        distr_1 = ot.Student(4,-2,1)
        distr_2 = ot.LogNormal(0,1,.5)
        distr_3 = ot.Triangular(1,3,5)
        
        left_distrs = [ot.Normal(2,1) for _ in range(dim-3)]
        
        target_distr = ot.ComposedDistribution([distr_1,distr_2,distr_3] + left_distrs)
        
        init_mean = ot.Point(np.zeros(dim))
        init_cov_matrix = ot.CovarianceMatrix(2*np.eye(dim)) 
        init_distr = ot.Normal(init_mean,init_cov_matrix)
        samples,log_W = adaptive_is(target_distr, init_distr, 10**4, 10,latent_dim=8)
        
        xx = np.linspace(-6,6,1001).reshape((-1,1))
        yy2 = np.array(init_distr.getMarginal(0).computePDF(xx)).flatten()
        
        last_sample = np.array(samples[-1])
        fig,ax = plt.subplots(2,5,figsize=(12,6))
        for i in range(2):
            for j in range(5):
                ax[i,j].hist(last_sample[:,5*i+j],bins=100,density=True)
                yy = np.array(target_distr.getMarginal(5*i+j).computePDF(xx)).flatten()
                ax[i,j].plot(xx,yy)
                ax[i,j].plot(xx,yy2)

    return fig,ax

st = time.time()
fig,ax = single_test(20)    
print(time.time()-st)        

#%% Test case 1

dim = 10

target_mean1 = ot.Point(2.5*np.ones(dim))
target_mean2 = ot.Point(-2.5*np.ones(dim))
target_cov_matrix = ot.CovarianceMatrix(np.eye(dim))

distrs = [ot.Normal(target_mean1,target_cov_matrix),ot.Normal(target_mean2,target_cov_matrix)]
target_distr_1 = ot.Mixture(distrs)

init_distr_1 = ot.Normal(dim)

n_rep = 100
N_1 = 10**4
# divergence_KL_1 = np.zeros(n_rep)
# samples_list_1 = []

#%%

n_start = 95

for n in range(n_start,n_rep):
    start = time.time()
    samples,vae = adaptive_is(target_distr_1, init_distr_1, N_1, 10, latent_dim=4)
    samples_list_1.append(samples[-1])
    divergence_KL_1[n] = compute_Dkl(target_distr_1,vae[0],vae[1],vae[2])
    print(f"Loop n°{n+1} done in {time.time() - start}")
    

#%% Save data

np.save("Data/generation_vae_10_dkl.npy",divergence_KL_1)
    
    
#%% Test case 2

dim = 20


distr_1 = ot.Student(4,-2,1)
distr_2 = ot.LogNormal(0,1,.5)
distr_3 = ot.Triangular(1,3,5)

target_mean = ot.Point(2*np.ones(dim-3))
target_cov_matrix = ot.CovarianceMatrix(np.eye(dim-3))
left_distrs = [ot.Normal(2,1) for _ in range(dim-3)]

target_distr_2 = ot.ComposedDistribution([distr_1,distr_2,distr_3] + left_distrs)

init_mean = ot.Point(np.zeros(dim))
init_cov_matrix = ot.CovarianceMatrix(2*np.eye(dim)) 
init_distr_2 = ot.Normal(init_mean,init_cov_matrix)


n_rep = 100
N_2 = 10**4
divergence_KL_2 = np.zeros(n_rep)

#%%

samples_list_2 = []
n_start = 0

for n in range(n_start,n_rep):
    start = time.time()
    samples,log_W = adaptive_is(target_distr_2, init_distr_2, N_2, 10, latent_dim=8)
    samples_list_2.append(samples)
    divergence_KL_2[n] = compute_Dkl(log_W[-1])
    print(f"Loop n°{n+1} done in {time.time() - start}")