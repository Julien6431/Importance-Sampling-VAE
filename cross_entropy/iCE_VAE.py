#%% Modules

import numpy as np
import scipy.optimize as spo
import openturns as ot
import sys
sys.path.append("../src/")
from VAE_IS_VP import fitted_vae
np.seterr(all='ignore')

"""
---------------------------------------------------------------------------
Improved cross entropy-based importance sampling with Single Gaussian
---------------------------------------------------------------------------
Created by:
Sebastian Geyer
Felipe Uribe
Iason Papaioannou
Daniel Straub

Assistant Developers:
Fong-Lin Wu
Matthias Willer
Peter Kaplan
Luca Sardi

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Version 2021-03:
* input dimensions of limit state function changed to rows*columns = 
  samples*dimensions
---------------------------------------------------------------------------
Comments:
* Adopt draft scripts from Sebastian and reconstruct the code to comply
  with the style of the published codes
* W is the original importance weight (likelihood ratio)
* W_t is the transitional importance weight of samples
* W_approx is the weight (ratio) between real and approximated indicator functions
---------------------------------------------------------------------------
Input:
* N         : number of samples per level
* g_fun     : limit state function
* max_it    : maximum number of iterations
* distr     : Nataf distribution object or marginal distribution object of the input variables
* CV_target : taeget correlation of variation of weights
---------------------------------------------------------------------------
Output:
* Pr        : probability of failure
* lv        : total number of levels
* N_tot     : total number of samples
* samplesU  : object with the samples in the standard normal space
* samplesX  : object with the samples in the original space
---------------------------------------------------------------------------
Based on:
1. Papaioannou, I., Geyer, S., & Straub, D. (2019).
   Improved cross entropy-based importance sampling with a flexible mixture model.
   Reliability Engineering & System Safety, 191, 106564
2. Geyer, S., Papaioannou, I., & Straub, D. (2019).
   Cross entropy-based importance sampling using Gaussian densities revisited. 
   Structural Safety, 76, 15–27
---------------------------------------------------------------------------
"""


#%% Algorithm

def iCEIS_VAE(N, phi, t, distr, CV_target, latent_dim=2, K=75):
    
    # Initialization of variables and storage
    N_tot = 0  # total number of samples
    max_it = 20
    sigma_t = np.zeros(max_it)


    if type(phi)==ot.func.Function:
        compute_output = lambda x : np.array(phi(x)) 
    else:
        compute_output = lambda x : phi(np.array(x)).reshape((-1,1))
    
    
    X = distr.getSample(N)
    samples = [X]
    Y = compute_output(X)
    N_tot += N
    
    sigma_t[0] = np.abs(10 * np.mean(Y))
    I = (Y>=t)
    
    cdf = compute_CDF((Y.flatten()-t) / sigma_t[0])
    
    #W_approx = np.divide(I, approx_normCDF((Y.flatten()-t) / sigma_t[0]))
    W_approx = np.divide(I, cdf.reshape((-1,1)))
    if np.sum(W_approx)>0:
        Cov_x = np.std(W_approx) / np.mean(W_approx)
        if Cov_x <= CV_target:
            return np.mean(I), N_tot, samples
    
    W_t = cdf.reshape((-1,1))

    # Function reference
    minimize = spo.fminbound

    # Iteration
    for j in range(1,max_it):
        
        vae,_,_ = fitted_vae(np.array(X).astype("float32"),W_t.astype("float32"),latent_dim,K,epochs=100,batch_size=100)
        
        X,log_gx = vae.getSample(N,with_pdf=True)
        log_fx = distr.computeLogPDF(X)
        log_W = log_fx - log_gx
        W = np.exp(log_W)
        samples.append(X)
        #f_X = np.array(distr.computePDF(X))
    
        #W = f_X/np.array(g_X)
    
        Y = compute_output(X)
        N_tot += N
        I = (Y >= t)
                
        #print(approx_normCDF((Y.flatten()-t) / sigma_t[j-1]))
        
        cdf = compute_CDF((Y.flatten()-t) / sigma_t[j-1])
        #print(np.min(cdf),np.min(approx_normCDF((Y.flatten()-t) / sigma_t[j-1])))
        
        #W_approx = np.divide(I, approx_normCDF((Y.flatten()-t) / sigma_t[j-1]).reshape((-1,1)))
        W_approx = np.divide(I, cdf.reshape((-1,1)))
        if np.sum(W_approx)>0:
            Cov_x = np.std(W_approx) / np.mean(W_approx)
            print(Cov_x)
            if Cov_x <= CV_target:
                break
        
        # compute sigma and weights for distribution fitting
        # minimize COV of W_t (W_t=approx_normCDF*W)
        # fmin = lambda x: abs(
        #     np.std(np.multiply(approx_normCDF((Y.flatten()-t) / x).reshape((-1,1)), W))
        #     / np.mean(np.multiply(approx_normCDF((Y.flatten()-t) / x).reshape((-1,1)), W))
        #     - CV_target
        # )
        fmin = lambda x: abs(
            np.std(np.multiply(compute_CDF((Y.flatten()-t) / x).reshape((-1,1)), W))
            / np.mean(np.multiply(compute_CDF((Y.flatten()-t) / x).reshape((-1,1)), W))
            - CV_target
        )
        sigma_new = minimize(fmin, 0, sigma_t[j-1])
        #print(sigma_new)
        sigma_t[j] = sigma_new

        # update W_t
        #W_t = np.multiply(approx_normCDF((Y.flatten()-t) / sigma_new).reshape((-1,1)), W)
        W_t = np.multiply(compute_CDF((Y.flatten()-t) / sigma_new).reshape((-1,1)), W)

    # Calculation of the Probability of failure
    Pr = 1 / N * sum(W[I])

    return Pr, samples, N_tot


def compute_CDF(xx):
    
    d = ot.Normal(1)
    yy = d.computeCDF(xx.reshape((-1,1)))
    return np.array(yy)


def approx_normCDF(x):
    # Returns an approximation for the standard normal CDF based on a
    # polynomial fit of degree 9

    erfun = np.zeros(len(x))

    idpos = x > 0
    idneg = x < 0

    t = (1 + 0.5 * abs(x / np.sqrt(2))) ** -1

    tau = t * np.exp(
        -((x / np.sqrt(2)) ** 2)
        - 1.26551223
        + 1.0000236 * t
        + 0.37409196 * (t ** 2)
        + 0.09678418 * (t ** 3)
        - 0.18628806 * (t ** 4)
        + 0.27886807 * (t ** 5)
        - 1.13520398 * (t ** 6)
        + 1.48851587 * (t ** 7)
        - 0.82215223 * (t ** 8)
        + 0.17087277 * (t ** 9)
    )
    erfun[idpos] = 1 - tau[idpos]
    erfun[idneg] = tau[idneg] - 1

    p = 0.5 * (1 + erfun)

    return p