"""
@author: Julien Demange-Chryst
"""


#%% Modules

import numpy as np
import openturns as ot
import matplotlib.pyplot as plt
import sys
sys.path.append("../src/")
from VAE_IS_VP import fitted_vae

#%% Setting

input_dim = 10
latent_dim = 2
K = 75
N = 10**4

X = np.array(ot.Normal(input_dim).getSample(N)).astype('float32')

y = np.abs(X[:,0])>1.5
y = y.reshape((-1,1)).astype('float32')

vae,vae_encoder,vae_decoder = fitted_vae(X,y,latent_dim,K,epochs=100)


#%% Plot latent space

mean_x = vae.mean_x
std_x = vae.std_x

X_normed = (X-mean_x)/std_x

fig,ax = plt.subplots(figsize=(12,9))
cm = plt.cm.get_cmap('cool')

pseudo_inputs = vae.get_pseudo_inputs()
_, _, z = vae_encoder(pseudo_inputs)
Z_mean,Z_log_var,Z = vae_encoder(X_normed)
im = ax.scatter(np.array(Z)[:,0],np.array(Z)[:,1],c=y,s=1,cmap=cm)
fig.colorbar(im, ax=ax)
ax.scatter(np.array(z)[:,0],np.array(z)[:,1],color='black',s=20)

fig.savefig(f"Figures/truncated_gaussian_latent_space_{input_dim}_{latent_dim}_{K}.png",bbox_inches='tight')

#%% New sample generation

distr1 = ot.TruncatedDistribution(ot.Normal(1), -1.5, ot.TruncatedDistribution.UPPER)
distr2 = ot.TruncatedDistribution(ot.Normal(1), 1.5, ot.TruncatedDistribution.LOWER)
failure_distr = ot.Mixture([distr1,distr2])

xx = np.linspace(-7,7,101).reshape((-1,1))
yy = failure_distr.computePDF(xx)
yyze = ot.Normal(1).computePDF(xx)

full = ot.ComposedDistribution([failure_distr] + (input_dim-1)*[ot.Normal(1)])

new_X = vae.getSample(10**5,with_pdf=False)

fig,ax = plt.subplots(2,5,figsize=(15,6))
for i in range(2):
    for j in range(5):
        ax[i,j].hist(np.array(new_X[:,5*i+j]),bins=100,density=True)
        if (i>0) or (j>0):
            ax[i,j].plot(xx,yyze)
ax[0,0].plot(xx,yy)

fig.savefig(f"Figures/truncated_gaussian_new_sample_{input_dim}_{latent_dim}_{K}.png",bbox_inches='tight')