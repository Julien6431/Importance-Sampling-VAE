#%% Modules

import numpy as np
import openturns as ot
import sys
sys.path.append("../src/")
from VAE_IS_VP import fitted_vae

import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf

#%% fzsd

def plot_latent_space(vae,X,y):
    
    Xnp = np.array(X).astype("float32")
    
    mean_x = vae.mean_x
    std_x = vae.std_x

    X_normed = (Xnp-mean_x)/std_x
    
    Xtf = tf.convert_to_tensor(X_normed)
    
    vae_encoder,vae_decoder = vae.get_encoder_decoder()
    
    fig,ax = plt.subplots(figsize=(10,9))
    
    pseudo_inputs = vae.get_pseudo_inputs()
    _, _, z = vae_encoder(pseudo_inputs)
    Z_mean,Z_log_var,Z = vae_encoder(Xtf)
    ax.scatter(np.array(Z)[:,0],np.array(Z)[:,1],c=y,s=2)
    ax.scatter(np.array(z)[:,0],np.array(z)[:,1],color='white',s=20)
    
    x_min,x_max = ax.get_xlim()
    y_min,y_max = ax.get_ylim()
    
    nb_points = 501
    
    x1 = np.linspace(x_min,x_max, nb_points)
    x2 = np.linspace(y_min,y_max, nb_points)
    
    X1, X2 = np.meshgrid(x1, x2)
    
    values_function = np.zeros((nb_points,nb_points))
    for i in range(nb_points):
        for j in range(nb_points):
            x = np.array([x1[j],x2[i]])
            values_function[i,j] = vae.prior.computePDF(x)
    
    ax.contourf(X1, X2, values_function,levels=200,cmap=cm.Reds)

    ax.set_xlabel("$z_1$",fontsize=20)
    ax.set_ylabel("$z_2$",fontsize=20)
    ax.tick_params(axis='x',labelsize='xx-large')
    ax.tick_params(axis='y',labelsize='xx-large')
    
    fig.savefig("Figures/latent_space_four_branches_100.png",bbox_inches='tight',dpi=500)
    
    return fig,ax


#%% Cross entropy with VAE

def CEIS_VAE(N,p,phi,t,distr,latent_dim=2,K=75,lat_space_plot=False):
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
    I = (Y>=gamma_hat[0])
    
    if gamma_hat[0]>=t:
        return np.mean(I),samples,N_tot
    
    WI = I
        
    #Adaptive algorithm
    for j in range(1,max_it):
                        
        vae,_,_ = fitted_vae(np.array(X)[I.flatten()].astype("float32"),WI[I.flatten()].astype("float32"),latent_dim,K,epochs=100,batch_size=100)
        
        if (latent_dim==2) and (lat_space_plot==True):
            fig,ax = plot_latent_space(vae, np.array(X)[I.flatten()].astype("float32"), WI[I.flatten()].astype("float32"))
            fig.show()
        
        X,log_gx = vae.getSample(N,with_pdf=True)
        log_fx = distr.computeLogPDF(X)
        log_W = log_fx - log_gx
        W = np.exp(log_W)
        samples.append(X)
    
        Y = compute_output(X)
        N_tot += N
        
        #Computation of the new threshold
        gamma_hat[j] = np.minimum(t,np.nanpercentile(Y,100*(1-p)))
                
        #Break the loop if the threshold is greater or equal to the real one
        if gamma_hat[j] >= t:
            break
        
        I = (Y>=gamma_hat[j])
        WI = I*W
        
    #Estimation of the failure probability  
    W_final = W
    I_final = (Y >= t)
    Pr = np.mean(I_final * W_final)

    return Pr, samples, N_tot
