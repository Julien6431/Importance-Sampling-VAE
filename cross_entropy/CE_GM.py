# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 08:57:59 2021

@author: jdemange
"""

#%% Modules

import numpy as np
import openturns as ot
from EMGM import EMGM

#%% Cross entropy with gaussian mixtures

def CEIS_GM(N,p,phi,t,distr,nb_mixture=2):
        
    dim = distr.getDimension()
    j = 0
    max_it = 50
    N_tot = 0
    
    if type(phi)==ot.func.Function:
        compute_output = lambda x : np.array(phi(x)).flatten()
    else:
        compute_output = lambda x : phi(np.array(x))
    
    mu_init = np.array(distr.getMean())*np.ones([1,dim])
    sigma_init = np.array(distr.getCovariance())[:,:,None]
    pi_init = np.array([1.0])
    gamma_hat = np.zeros(max_it+1)
    
    mu_hat = mu_init
    sigma_hat = sigma_init
    pi_hat = pi_init
    gamma_hat[0]= -np.inf
    
    samples = []
        
    for j in range(max_it):
        if len(pi_hat) == 1:
            aux_distr = ot.Normal(ot.Point(mu_hat[0]),ot.CovarianceMatrix(sigma_hat[:,:,0]))
        else:
            collDist = [ot.Normal(ot.Point(mu_hat[i]),ot.CovarianceMatrix(sigma_hat[:,:,i])) for i in range(mu_hat.shape[0])]
            aux_distr = ot.Mixture(collDist,pi_hat)
            
            
        ot_X = aux_distr.getSample(N)
        X = np.array(ot_X)
        
        samples.append(X)
        
        N_tot += N
        
        Y = compute_output(X)     
        h = np.array(aux_distr.computePDF(X))
                
        if gamma_hat[j] >= t:
            break
        
        gamma_hat[j+1] = np.minimum(t,np.nanpercentile(Y,100*(1-p)))
        I = (Y>=gamma_hat[j+1])
        W = np.array(distr.computePDF(X))/h
        W = W.flatten()
        
        [mu_hat, sigma_hat, pi_hat] = EMGM(X[I, :].T, W[I], nb_mixture)
        mu_hat = mu_hat.T
        
    lv = j
    gamma_hat = gamma_hat[: lv + 1]
        
    W_final = np.array(distr.computePDF(X))/h 
    W_final = W_final.flatten()
    I_final = (Y >= t)
    Pr = 1 / N * sum(I_final * W_final)
    
    return Pr, samples, aux_distr, N_tot
