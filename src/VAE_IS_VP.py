#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 12:04:51 2023

@author: juliendemange-chryst
"""


#%% Modules

import numpy as np
import tensorflow as tf
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from tensorflow import keras
from tensorflow.keras import layers
from AE_class import AutoEncoder
from VAE_class import VAE

#%% Initial VampPrior

def create_pseudo_inputs_layer(input_dim,K):
    id_points = keras.Input(shape=(K,))
    x = layers.Dense(128, activation="linear")(id_points)
    pseudo_input = layers.Dense(input_dim,activation="linear")(x)
    dense_layer = keras.Model(id_points, pseudo_input, name="dense_pseudo_inputs")
    return dense_layer

def initial_vp_layer(K,X,y):
    
    N,input_dim = X.shape
    dense_layer = create_pseudo_inputs_layer(input_dim,K)
    
    prob = y/np.sum(y)
    
    id_matrix = tf.eye(K)
    idx = np.random.choice(range(N),replace=False,size=K,p=prob.flatten())
    target_points = tf.convert_to_tensor(X[idx])
    dense_layer.compile(optimizer=keras.optimizers.Adam(),loss='mse')
    dense_layer.fit(id_matrix,target_points, epochs=100, batch_size=K,verbose=0)
    return dense_layer

#%% Initial autoencoder

def fitted_ae(X,y,latent_dim,epochs,batch_size):
    input_dim = X.shape[1]
    ae = AutoEncoder(input_dim, latent_dim)
    ae.compile(optimizer=keras.optimizers.Adam())
    ae.fit(tf.convert_to_tensor(X),tf.convert_to_tensor(y), epochs=epochs, batch_size=batch_size,verbose=0)
    return ae


#%% Fitted VAE

def fitted_vae(X,y,latent_dim,K,epochs=100,batch_size=100):
    input_dim = X.shape[1]
    
    vp_layer = initial_vp_layer(K,X,y)
    
    auto_encoder = fitted_ae(X,y,latent_dim,epochs=epochs,batch_size=batch_size)
    encoder,decoder = auto_encoder.get_encoder_decoder()
    
    vae = VAE(encoder, decoder, vp_layer, input_dim, latent_dim, K)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(tf.convert_to_tensor(X),tf.convert_to_tensor(y), epochs=epochs, batch_size=batch_size,verbose=0)
    vae.set_ot_prior()    
    vae_encoder,vae_decoder = vae.get_encoder_decoder()
    return vae,vae_encoder,vae_decoder
