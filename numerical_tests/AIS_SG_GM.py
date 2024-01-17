#%% Modules

import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
from EMGM import EMGM

#%%


def compute_Dkl(target_distr,simu_distr):
    
    sample = target_distr.getSample(5*10**4)
    log_target = np.array(target_distr.computeLogPDF(sample)).flatten()
    log_simu = np.array(simu_distr.computeLogPDF(sample)).flatten()
        
    return np.mean(log_target - log_simu)


def ais_sg_gm(target_distr,init_distr,N,max_iter,aux_family='SG'):
    
    dim = target_distr.getDimension()
    
    X = init_distr.getSample(N)
    N_tot = N
    
    log_target = target_distr.computeLogPDF(X)
    log_init = init_distr.computeLogPDF(X)
    
    W = np.exp(log_target-log_init)
    
    samples = X
    aux_distrs = [init_distr]
    
    for n in range(max_iter-1):    
        if aux_family == 'SG':
    
            mu_hat = np.mean(W.reshape((-1,1))*np.array(X),axis=0) / np.mean(W)
            Xtmp = np.array(X) - mu_hat
            Xo = np.sqrt(W).reshape((-1,1)) * (Xtmp)
            
            sigma_hat = np.matmul(Xo.T, Xo) / np.sum(W) + 1e-6 * np.eye(dim)
            
            aux_distr = ot.Normal(ot.Point(mu_hat),ot.CovarianceMatrix(sigma_hat))
            
        elif aux_family == 'GM':
            [mu_hat, sigma_hat, pi_hat] = EMGM(np.array(X).T, W, 2)
            mu_hat = mu_hat.T
                    
            collDist = [ot.Normal(ot.Point(mu_hat[i]),ot.CovarianceMatrix(sigma_hat[:,:,i])) for i in range(mu_hat.shape[0])]
            aux_distr = ot.Mixture(collDist,pi_hat)
            
        X = aux_distr.getSample(N)
        N_tot += N
        
        log_target = target_distr.computeLogPDF(X)
        log_aux = aux_distr.computeLogPDF(X)
        
        W = np.exp(log_target-log_aux)
        
        samples.add(X)
        aux_distrs.append(aux_distr)
        
    final_distr = aux_distr
    return X,W,final_distr



#%% Single test

def single_test(dim):
    if dim == 10:
        target_mean1 = ot.Point(2.5*np.ones(dim))
        target_mean2 = ot.Point(-2.5*np.ones(dim))
        
        target_cov_matrix = ot.CovarianceMatrix(np.eye(dim)) 
        
        distrs = [ot.Normal(target_mean1,target_cov_matrix),ot.Normal(target_mean2,target_cov_matrix)]

        target_distr = ot.Mixture(distrs)
        init_mean = ot.Point(np.zeros(dim))
        init_cov_matrix = ot.CovarianceMatrix(np.eye(dim)) 
        init_distr = ot.Normal(init_mean,init_cov_matrix)
        samples,W,final_distr = ais_sg_gm(target_distr, init_distr, 10**4, 30,aux_family='GM')
        
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
        distr_1 = ot.Student(4,2,1)
        distr_2 = ot.LogNormal(0,1)
        distr_3 = ot.Triangular(1,3,5)
        
        left_distrs = [ot.Normal(2,1) for _ in range(dim-3)]
        
        target_distr = ot.ComposedDistribution([distr_1,distr_2,distr_3] + left_distrs)
        
        init_mean = ot.Point(0*np.ones(dim))
        init_cov_matrix = ot.CovarianceMatrix(2*np.eye(dim)) 
        init_distr = ot.Normal(init_mean,init_cov_matrix)
        samples,W,_ = ais_sg_gm(target_distr, init_distr, 10**4, 20)
        
        xx = np.linspace(-6,6,1001).reshape((-1,1))
        yy2 = np.array(init_distr.getMarginal(0).computePDF(xx)).flatten()
        
        last_sample = np.array(samples)
        fig,ax = plt.subplots(2,5,figsize=(12,6))
        for i in range(2):
            for j in range(5):
                ax[i,j].hist(last_sample[:,5*i+j],bins=100,density=True,weights=W)
                yy = np.array(target_distr.getMarginal(5*i+j).computePDF(xx)).flatten()
                ax[i,j].plot(xx,yy)
                ax[i,j].plot(xx,yy2)
                
    return fig,ax

fig,ax = single_test(20)    

#%% Test case dimension 10

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
    samples,_,final_distr = ais_sg_gm(target_distr_1, init_distr_1, N_1, 10,aux_family='GM')
    samples_list_1.append(samples)
    divergence_KL_1[n] = compute_Dkl(target_distr_1,final_distr)
    

#%% Save data
    
import pickle

fileObj = open('Data/generation_aisGM_10.pkl', 'wb')
pickle.dump(samples_list_1,fileObj)
fileObj.close()

np.save("Data/generation_aisGM_10_dkl.npy",divergence_KL_1)
