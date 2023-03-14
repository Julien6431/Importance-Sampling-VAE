# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:08:10 2022

@author: Julien Demange-Chryst
"""

#%% Modules

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
    
#%% Encoder/Decoder architecture
    
def create_encoder(input_dim,latent_dim):
    
    encoder_inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(32, activation="linear")(encoder_inputs)
    x = layers.Dense(16, activation="linear")(x)
    z = layers.Dense(latent_dim, activation="linear", name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, activation="linear", name="z_log_var")(x)
    encoder = keras.Model(encoder_inputs, [z,z_log_var], name="encoder")
    return encoder

def create_decoder(input_dim,latent_dim):
    
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(16, activation="linear")(latent_inputs)
    x = layers.Dense(32, activation="linear")(x)
    x_output = layers.Dense(input_dim, activation="linear")(x)
    x_log_var = layers.Dense(input_dim, activation="linear")(x)
    decoder = keras.Model(latent_inputs, [x_output,x_log_var], name="decoder")
    return decoder


#%% Autoencoder class

class AutoEncoder(keras.Model):
    def __init__(self, input_dim, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = create_encoder(input_dim,latent_dim)
        self.decoder = create_decoder(input_dim,latent_dim)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker]

    def get_encoder_decoder(self):
        return self.encoder,self.decoder

    def train_step(self, data):
        
        X,y = data
        y = tf.reshape(y,[-1])      
    
        with tf.GradientTape() as tape:
            
            z,_ = self.encoder(X)
            reconstruction,_ = self.decoder(z)
            
            xx = tf.pow(X - reconstruction,2)
            total_loss = tf.reduce_mean(tf.multiply(tf.reduce_mean(xx,axis=1),y))
            
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        
        return {"loss": self.total_loss_tracker.result()}