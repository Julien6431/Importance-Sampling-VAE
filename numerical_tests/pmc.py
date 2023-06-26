# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:02:33 2023

@author: jdemange
"""


#%% Modules

import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
import time

#%%

def compute_Dkl(target_distr,simu_distr):
    
    sample = target_distr.getSample(5*10**4)
    log_target = np.array(target_distr.computeLogPDF(sample)).flatten()
    log_simu = np.array(simu_distr.computeLogPDF(sample)).flatten()
    
    return np.mean(log_target - log_simu)


def pmc(target_distr,sigma2,N,max_iter):
    
    dim = target_distr.getDimension()

    cov_matrix = sigma2*np.eye(dim)
    # init_distr = ot.Normal(ot.Point(dim),ot.CovarianceMatrix(cov_matrix))
        
    # X = init_distr.getSample(N)
    # log_target = target_distr.computeLogPDF(X)
    # log_init = init_distr.computeLogPDF(X)
    
    X = ot.Sample(N,dim)
    samples = []
    
    for n in range(max_iter):
        print(n)
        iter_x = ot.Sample(1,dim)
        w = ot.Sample()
        mixture = []
        for point in X:
            d = ot.Normal(point,ot.CovarianceMatrix(cov_matrix)) 
            mixture.append(d)
            new_point = d.getSample(1)
            iter_x.add(new_point)

            
            log_target = target_distr.computeLogPDF(new_point)
            log_distr = d.computeLogPDF(new_point)
            
            w.add(np.exp(log_target - log_distr))
        
        if n==max_iter-1:
            return iter_x,ot.Mixture(mixture)
                
        w_np = np.array(w).flatten()
        w_np = w_np/np.sum(w_np)
        indices = np.random.choice(range(1,N+1),size=N,replace=True,p=w_np)
        X = iter_x[indices]
        samples.append(X)
    
    final_distr = ot.Mixture(mixture,w_np)
    
    return samples,final_distr
    
    
    # X = init_distr.getSample(N)
    # log_targetX = target_distr.computeLogPDF(X)
    # log_initX = init_distr.computeLogPDF(X)
    
    # W = np.exp(log_targetX - log_initX)
    # samples = [X]
    # log_W = [log_targetX - log_initX]
    
    # for n in range(max_iter):
    #     #print(n)
    #     vae,_,_ = fitted_vae(np.array(X).astype('float32'), W.astype('float32'), latent_dim=latent_dim, K=75)
        
    #     X,log_gX = vae.getSample(N,with_pdf=True)
    #     log_targetX = target_distr.computeLogPDF(X)
    #     W = np.exp(log_targetX-log_gX)
        
    #     samples.append(X)
    #     log_W.append(log_targetX-log_gX)
    
    # return samples,log_W


    
    
#%% Single test


def single_test(dim):
    if dim == 10:
        target_mean1 = ot.Point(2.5*np.ones(dim))
        target_mean2 = ot.Point(-2.5*np.ones(dim))
        
        target_cov_matrix = ot.CovarianceMatrix(np.eye(dim)) 
        
        distrs = [ot.Normal(target_mean1,target_cov_matrix),ot.Normal(target_mean2,target_cov_matrix)]

        target_distr = ot.Mixture(distrs)
        #target_distr = ot.Normal(target_mean1,target_cov_matrix)
        init_mean = ot.Point(np.zeros(dim))
        init_cov_matrix = ot.CovarianceMatrix(1*np.eye(dim)) 
        init_distr = ot.Normal(init_mean,init_cov_matrix)
        samples,final_distr = pmc(target_distr, 1, 10**4, 10)
        

        xx = np.linspace(-6,6,1001).reshape((-1,1))
        yy = np.array(target_distr.getMarginal(0).computePDF(xx)).flatten()
        yy2 = np.array(init_distr.getMarginal(0).computePDF(xx)).flatten()
                
        last_sample = np.array(samples)#[-1])
        fig,ax = plt.subplots(2,5,figsize=(12,6))
        for i in range(2):
            for j in range(5):
                ax[i,j].hist(last_sample[:,5*i+j],bins=100,density=True)
                ax[i,j].plot(xx,yy)
                ax[i,j].plot(xx,yy2)
                #yy3 = np.array(final_distr.getMarginal(5*i+j).computePDF(xx)).flatten()
                #ax[i,j].plot(xx,yy3)
               
        print("b")
        #print(compute_Dkl(target_distr,final_distr))
                
    elif dim == 20:
        distr_1 = ot.Student(4,-2,1)
        distr_2 = ot.LogNormal(0,1,.5)
        distr_3 = ot.Triangular(1,3,5)
        
        left_distrs = [ot.Normal(2,1) for _ in range(dim-3)]
        
        target_distr = ot.ComposedDistribution([distr_1,distr_2,distr_3] + left_distrs)
        
        init_mean = ot.Point(np.zeros(dim))
        init_cov_matrix = ot.CovarianceMatrix(2*np.eye(dim)) 
        init_distr = ot.Normal(init_mean,init_cov_matrix)
        samples,final_distr = pmc(target_distr, .25, 10**4, 20)
        
        xx = np.linspace(-6,6,1001).reshape((-1,1))
        yy2 = np.array(init_distr.getMarginal(0).computePDF(xx)).flatten()
        
        last_sample = np.array(samples)
        fig,ax = plt.subplots(2,5,figsize=(12,6))
        for i in range(2):
            for j in range(5):
                ax[i,j].hist(last_sample[:,5*i+j],bins=100,density=True)
                yy = np.array(target_distr.getMarginal(5*i+j).computePDF(xx)).flatten()
                ax[i,j].plot(xx,yy)
                ax[i,j].plot(xx,yy2)
                
        print(compute_Dkl(target_distr,final_distr))

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
divergence_KL_1 = np.zeros(n_rep)
samples_list_1 = []

#%%

n_start = 0

for n in range(n_start,n_rep):
    start = time.time()
    samples,final_distr = pmc(target_distr_1, 1.25, N_1, 10)
    samples_list_1.append(samples[-1])
    divergence_KL_1[n] = compute_Dkl(target_distr_1,final_distr)
    print(f"Loop n°{n+1} done in {time.time() - start}")
    

#%% Save data
    
np.save("Data/generation_pmc_10_dkl.npy",divergence_KL_1)
    
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