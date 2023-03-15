# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 14:00:16 2021

@author: jdemange
"""

#%% Modules

import numpy as np
import openturns as ot

#%% Cross entropy with gaussian

def CEIS_SG(N,p,phi,t,distr):
    if (N * p != np.fix(N * p)) or (1 / p != np.fix(1 / p)):
        raise RuntimeError(
            "N*p and 1/p must be positive integers. Adjust N and p accordingly"
        )
    
    j = 0
    max_it = 50
    N_tot = 0
    
    if type(phi)==ot.func.Function:
        compute_output = lambda x : np.array(phi(x)).flatten()
    else:
        compute_output = lambda x : phi(np.array(x))
    
    dim = distr.getDimension()
    
    mu_init = np.array(distr.getMean())
    sigma_init = np.diag(np.diag(np.array(distr.getCovariance())))
    gamma_hat = np.zeros(max_it+1)
    
    mu_hat = mu_init
    sigma_hat = sigma_init
    gamma_hat[0]= -np.inf
    
    samples = []
        
    for j in range(max_it):
        ot_mu = ot.Point(mu_hat)
        ot_sigma = ot.CovarianceMatrix(sigma_hat)
        aux_distr = ot.Normal(ot_mu,ot_sigma)
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
        
        prod = np.matmul(W[I], X[I, :])
        summ = np.sum(W[I])
        mu_hat = (prod) / summ
        Xtmp = X[I, :] - mu_hat
        Xo = (Xtmp) * np.tile(np.sqrt(W[I]), (dim, 1)).T
        sigma_hat = np.matmul(Xo.T, Xo) / np.sum(W[I]) + 1e-6 * np.eye(dim)
        
    lv = j
    gamma_hat = gamma_hat[: lv + 1]

    # Calculation of the Probability of failure
    W_final = np.array(distr.computePDF(X))/h
    W_final = W_final.flatten()
    I_final = (Y >= t)
    Pr = 1 / N * sum(I_final * W_final)

    return Pr, samples, aux_distr, N_tot