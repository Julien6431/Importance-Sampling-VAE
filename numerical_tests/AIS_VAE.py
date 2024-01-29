#%% Modules

import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
import tensorflow.keras.backend as kb
import sys
sys.path.append("src")
from src.VAE_IS_VP import fitted_vae


#%%


def compute_Dkl(target_distr,simu_distr,mean_x,std_x):
    
    sample = target_distr.getSample(5*10**4)
    log_target = np.array(target_distr.computeLogPDF(sample)).flatten()
    
    sample_np = np.array(sample)
    sample_std = ot.Sample((sample_np - mean_x)/std_x)
        
    log_simu = np.array(simu_distr.computeLogPDF(sample_std)).flatten() - np.sum(np.log(std_x))
    
    return np.mean(log_target - log_simu)
    

def adaptive_is_vae(target_distr,init_distr,N,max_iter,latent_dim=2):
        
    X = init_distr.getSample(N)
    log_targetX = target_distr.computeLogPDF(X)
    log_initX = init_distr.computeLogPDF(X)
    
    W = np.exp(log_targetX - log_initX)
    samples = [X]
    
    for n in range(max_iter):
        kb.clear_session()
        vae,_,_ = fitted_vae(np.array(X).astype('float32'), W.astype('float32'), latent_dim=latent_dim, K=75)
        
        X,log_gX = vae.getSample(N,with_pdf=True)
        log_targetX = target_distr.computeLogPDF(X)
        W = np.exp(log_targetX-log_gX)
        
        samples.append(X)
    
        mean_x = vae.mean_x.astype('float64')
        std_x = vae.std_x.astype('float64')
        distrX = vae.distrX
     
         
    return X,W,[distrX,mean_x,std_x]

    
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
        samples,W,vae = adaptive_is_vae(target_distr, init_distr, 10**4, 5,latent_dim=4)


        xx = np.linspace(-6,6,1001).reshape((-1,1))
        yy = np.array(target_distr.getMarginal(0).computePDF(xx)).flatten()
        yy2 = np.array(init_distr.getMarginal(0).computePDF(xx)).flatten()
        
        last_sample = np.array(samples)
        fig,ax = plt.subplots(2,5,figsize=(12,6))
        for i in range(2):
            for j in range(5):
                ax[i,j].hist(last_sample[:,5*i+j],bins=100,density=True,weights=W)
                ax[i,j].plot(xx,yy)
                ax[i,j].plot(xx,yy2)
                
                
    elif dim == 20:
        distr_1 = ot.Student(4,-2,1)
        distr_2 = ot.LogNormal(0,1)
        distr_3 = ot.Triangular(1,3,5)
        
        left_distrs = [ot.Normal(2,1) for _ in range(dim-3)]
        
        target_distr = ot.ComposedDistribution([distr_1,distr_2,distr_3] + left_distrs)
        
        init_mean = ot.Point(np.zeros(dim))
        init_cov_matrix = ot.CovarianceMatrix(2*np.eye(dim)) 
        init_distr = ot.Normal(init_mean,init_cov_matrix)
        sample,W,vae = adaptive_is_vae(target_distr, init_distr, 10**4, 10,latent_dim=8)
        
        xx = np.linspace(-6,6,1001).reshape((-1,1))
        yy2 = np.array(init_distr.getMarginal(0).computePDF(xx)).flatten()
        
        last_sample = np.array(sample)
        fig,ax = plt.subplots(2,5,figsize=(12,6))
        for i in range(2):
            for j in range(5):
                ax[i,j].hist(last_sample[:,5*i+j],bins=100,density=True,weights=W)
                yy = np.array(target_distr.getMarginal(5*i+j).computePDF(xx)).flatten()
                ax[i,j].plot(xx,yy)
                ax[i,j].plot(xx,yy2)

    dkl = compute_Dkl(target_distr,vae[0],vae[1],vae[2])
    print(dkl)

    return fig,ax

fig,ax = single_test(10)    

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

for n in range(n_rep):
    samples,W,vae = adaptive_is_vae(target_distr_1, init_distr_1, N_1, 10, latent_dim=4)
    samples_list_1.append(samples[-1])
    divergence_KL_1[n] = compute_Dkl(target_distr_1,vae[0],vae[1],vae[2])
    

#%% Save data

import pickle

fileObj = open('Data/generation_vae_10.pkl', 'wb')
pickle.dump(samples_list_1,fileObj)
fileObj.close()

np.save("Data/generation_vae_10_dkl.npy",divergence_KL_1)
    
#%% Test case 2

dim = 20


distr_1 = ot.Student(4,-2,1)
distr_2 = ot.LogNormal(0,1)
distr_3 = ot.Triangular(1,3,5)

left_distrs = [ot.Normal(2,1) for _ in range(dim-3)]

R = ot.CorrelationMatrix(dim)
for i in range(dim-1):
    R[i, i+1] = 0.25
copula = ot.NormalCopula(R)

target_distr_2 = ot.ComposedDistribution([distr_1,distr_2,distr_3] + left_distrs,copula)

init_mean = ot.Point(np.zeros(dim))
init_cov_matrix = ot.CovarianceMatrix(2*np.eye(dim)) 
init_distr_2 = ot.Normal(init_mean,init_cov_matrix)


n_rep = 10
N_2 = 10**4
divergence_KL_2 = np.zeros(n_rep)
samples_list_2 = []
weights_2 = []

for n in range(n_rep):
    samples,W,vae = adaptive_is_vae(target_distr_2, init_distr_2, N_2, 10, latent_dim=8)
    samples_list_2.append(samples)
    weights_2.append(W)
    divergence_KL_2[n] = compute_Dkl(target_distr_2,vae[0],vae[1],vae[2])
        
        
#%% Save data

import pickle

fileObj = open('Data/generation_vae_20.pkl', 'wb')
pickle.dump(samples_list_2,fileObj)
fileObj.close()

fileObj = open('Data/generation_vae_20_weights.pkl', 'wb')
pickle.dump(weights_2,fileObj)
fileObj.close()
