# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:08:10 2022

@author: Julien Demange-Chryst
"""

#%% Modules

import numpy as np
import openturns as ot
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
tfd = tfp.distributions

#%% Sampling procedure

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
#%% Encoder/Decoder architecture
    
def create_encoder(encoder,input_dim,latent_dim):
    
    encoder_inputs = keras.Input(shape=(input_dim,))
    z_mean,z_log_var = encoder(encoder_inputs)
    z = Sampling()([z_mean, z_log_var])
    encoder_vae = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder_vae

def create_decoder(decoder,input_dim,latent_dim):
    
    latent_inputs = keras.Input(shape=(latent_dim,))
    x_mean,x_log_var = decoder(latent_inputs)
    decoder_vae = keras.Model(latent_inputs, [x_mean, x_log_var], name="decoder")
    return decoder_vae


#%% VAE class

class VAE(keras.Model):
    def __init__(self, encoder, decoder, vp_layer, input_dim, latent_dim, K, mean_x,std_x, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.K = K
        self.encoder = create_encoder(encoder,input_dim,latent_dim)
        self.decoder = create_decoder(decoder,input_dim,latent_dim)
        self.pseudo_inputs_layer = vp_layer
        self.prior = ot.Normal(latent_dim)
        self.distrX = ot.Normal(input_dim)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
        self.mean_x = mean_x
        self.std_x = std_x

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def get_encoder_decoder(self):
        return self.encoder,self.decoder
    
 
    def mixture(self,z_mean,z_log_var):
        components = []
        cat = tfd.Categorical(probs=self.K*[1/self.K])
        for k in range(self.K):
            mu = z_mean[k]
            sigma = tf.exp(z_log_var[k])
            d_k = tfp.distributions.MultivariateNormalDiag(loc=mu,scale_diag=tf.sqrt(sigma))
            components.append(d_k)
        return tfd.Mixture(cat = cat,components = components)
    

    def train_step(self, data):
        X,y = data
        y = tf.reshape(y,[-1])
        
        id_matrix = tf.eye(self.K)
        N_01_x = tfd.MultivariateNormalDiag(loc=self.input_dim*[0],scale_diag=self.input_dim*[1])
        N_01_latent = tfd.MultivariateNormalDiag(loc=self.latent_dim*[0],scale_diag=self.latent_dim*[1])
        
    
        with tf.GradientTape() as tape:
            
            pseudo_inputs = self.pseudo_inputs_layer(id_matrix)
            z_mean_ps, z_log_var_ps,_ = self.encoder(pseudo_inputs)
            prior_distr = self.mixture(z_mean_ps,z_log_var_ps)
            
        
            z_mean, z_log_var, z = self.encoder(X)
            reconstruction, log_var = self.decoder(z)
            
            x_scale = tf.sqrt(tf.exp(log_var))
            xx = (X-reconstruction)/x_scale
            
            reconstruction_loss = -1/2*tf.reduce_sum(log_var,axis=1) + N_01_x.log_prob(xx)
            reconstruction_loss = -tf.reduce_mean(tf.multiply(reconstruction_loss,y))
            
            log_pdf_prior = prior_distr.log_prob(z)
            
            z_var_inv = 1/tf.exp(z_log_var)            
            n_01 = tf.sqrt(z_var_inv)*(z-z_mean)
            
            log_pdf_post = -0.5*tf.reduce_sum(z_log_var, axis=1) + N_01_latent.log_prob(n_01)
                        
            kl_loss = tf.reduce_mean(tf.multiply(log_pdf_post - log_pdf_prior,y))
            
            total_loss = (reconstruction_loss + kl_loss)
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }
    
    def get_pseudo_inputs(self):
        return self.pseudo_inputs_layer(tf.eye(self.K))
    
    def set_ot_prior(self):
        pseudo_inputs = self.get_pseudo_inputs()

        z_mean, z_log_var, z = self.encoder(pseudo_inputs)
        z_mean = ot.Sample(np.array(z_mean).astype("float64"))
        z_var = np.exp(np.array(z_log_var).astype("float64"))
        z_std = ot.Sample(np.sqrt(z_var))        
        
        distrs = []
        for i in range(self.K):
            d  = ot.Normal(z_mean[i],z_std[i])
            distrs.append(d)
        prior = ot.Mixture(distrs)
        self.prior = prior
        
    def set_ot_distrX(self,M):
        Z = self.prior.getSample(M)
        X_mean, X_log_var = self.decoder(tf.convert_to_tensor(Z))
        X_mean = ot.Sample(np.array(X_mean).astype("float64"))
        X_log_var = np.array(X_log_var).astype("float64")
        X_std = ot.Sample(np.sqrt(np.exp(X_log_var)))
        
        distrs = []
        for i in range(M):
            d = ot.Normal(X_mean[i],X_std[i])
            distrs.append(d)
        distrX = ot.Mixture(distrs)
        self.distrX = distrX
    
    
    def getSample(self,N,with_pdf=False):
        new_sample_std = np.array(self.distrX.getSample(N))
        new_sample = ot.Sample(self.mean_x + self.std_x*new_sample_std)
        if with_pdf==True:
            det = np.prod(self.std_x)
            g_X = self.distrX.computePDF(new_sample_std)/det
            return new_sample,g_X
        else:
            return new_sample

            
    # def getSample(self,N,with_pdf=False):
    #     z_sample = self.prior.getSample(N)
    #     X_mean, X_log_var = self.decoder(tf.convert_to_tensor(z_sample))
    #     X_mean,X_log_var = ot.Sample(np.array(X_mean).astype("float64")),np.array(X_log_var).astype("float64")
    
    #     std_matrix = np.sqrt(np.exp(X_log_var))
    
    #     Normal_vector = ot.Normal(self.input_dim).getSample(N)
    #     new_sample = ot.Sample(std_matrix*Normal_vector) + X_mean
        
    #     if with_pdf==True:
    #         start_time = time.time()
    #         new_sample_np = np.array(new_sample)
    #         g_X = np.zeros((N,1))
    #         log_det = 0.5*np.sum(X_log_var,axis=1)
    #         inv_det = 1/np.exp(log_det)
            
    #         for i in range(N):
    #             point = (new_sample_np[i] - X_mean)/std_matrix
    #             pdf = np.array(ot.Normal(self.input_dim).computePDF(ot.Sample(point))).flatten()
    #             g_X[i] = np.mean(inv_det*pdf)
       
    #         print("Get PDF time: %s seconds " % (time.time() - start_time))
    #
    #         return new_sample, ot.Sample(g_X)
    #     else:
    #         return new_sample