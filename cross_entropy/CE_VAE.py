#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 18:42:19 2023

@author: juliendemange-chryst
"""

#%% Modules

import numpy as np
import openturns as ot
import sys
sys.path.append("../src/")
from VAE_IS_VP import fitted_vae

import matplotlib.pyplot as plt
import tensorflow as tf

#%% fzsd

def plot_latent_space(vae,X,y):
    
    Xnp = np.array(X).astype("float32")
    Xtf = tf.convert_to_tensor(Xnp)
    
    vae_encoder,vae_decoder = vae.get_encoder_decoder()
    
    fig,ax = plt.subplots(figsize=(12,9))
    cm = plt.cm.get_cmap('cool')
    
    pseudo_inputs = vae.get_pseudo_inputs()
    _, _, z = vae_encoder(pseudo_inputs)
    Z_mean,Z_log_var,Z = vae_encoder(Xtf)
    im = ax.scatter(np.array(Z)[:,0],np.array(Z)[:,1],c=y,s=1,cmap=cm)
    fig.colorbar(im, ax=ax)
    ax.scatter(np.array(z)[:,0],np.array(z)[:,1],color='black',s=20)
    
    return fig,ax


#%% Cross entropy with VAE

def CEIS_VAE(N,p,phi,t,distr,latent_dim=2,K=75):
    """
    

    Parameters
    ----------
    N : INT
        Number of points drawn at each step.
    p : FLOAT
        Quantile to set the new intermediate critical threshold, 0<=p<=1.
    phi : FUNCTION
        Function to estimate the failure probability.
    t : FLOAT
        Failure threshold.
    distr : OPENTURS DISTRIBUTION
        Input distribution.
    latent_dim : INT, optional
        Dimension of the latent space. The default is 2.
    K : INT, optional
        Number of components for the VampPrior. The default is 75.

    Raises
    ------
    RuntimeError
        N*p and 1/p must be positive integers. Adjust N and p accordingly.

    Returns
    -------
    Pr : FLOAT
        Estimated failure probability.
    samples : LIST
        List containing the samples drawn at each iteration.
    N_tot : INT
        Total number of calls to the function.

    """
    
    # if (N * p != np.fix(N * p)) or (1 / p != np.fix(1 / p)):
    #     raise RuntimeError(
    #         "N*p and 1/p must be positive integers. Adjust N and p accordingly"
    #     )
    
    j = 0 #current iteration
    max_it = 50 #maximal number of iterations
    N_tot = 0 #number of calls to the function
    gamma_hat = np.zeros(max_it)
    
   
    if type(phi)==ot.func.Function:
        compute_output = lambda x : np.array(phi(x)) 
    else:
        compute_output = lambda x : phi(np.array(x)).reshape((-1,1))
    
    
    #Initialisation
    X = distr.getSample(N)
    samples = [X]
    Y = compute_output(X)
    N_tot += N
    
    gamma_hat[0] = np.minimum(t,np.nanpercentile(Y,100*(1-p)))
    #print(gamma_hat[0])
    I = (Y>=gamma_hat[0])
    
    if gamma_hat[0]>=t:
        return np.mean(I),samples,N_tot
    
    WI = I
        
    #Adaptive algorithm
    for j in range(1,max_it):
                        
        vae,_,_ = fitted_vae(np.array(X)[I.flatten()].astype("float32"),WI[I.flatten()].astype("float32"),latent_dim,K,epochs=100,batch_size=100)
        #vae,_,_ = fitted_vae(np.array(X).astype("float32"),W.astype("float32"),latent_dim,K,epochs=100,batch_size=100)
        
        #fig,ax = plot_latent_space(vae, np.array(X)[I.flatten()].astype("float32"), WI[I.flatten()].astype("float32"))
        #fig.show()
        
        #X,g_X = vae.getSample(N,with_pdf=True)
        X,log_gx = vae.getSample(N,with_pdf=True)
        log_fx = distr.computeLogPDF(X)
        log_W = log_fx - log_gx
        W = np.exp(log_W)
        samples.append(X)
        #f_X = np.array(distr.computePDF(X))
    
        Y = compute_output(X)
        N_tot += N
        
        #Computation of the new threshold
        gamma_hat[j] = np.minimum(t,np.nanpercentile(Y,100*(1-p)))
                
        #print(gamma_hat[j])
        #Break the loop if the threshold is greater or equal to the real one
        if gamma_hat[j] >= t:
            break
        
        I = (Y>=gamma_hat[j])
        WI = I*W#f_X/np.array(g_X)

    #Estimation of the failure probability  
    #W_final = f_X/np.array(g_X)
    W_final = W
    I_final = (Y >= t)
    Pr = np.mean(I_final * W_final)

    return Pr, samples, N_tot