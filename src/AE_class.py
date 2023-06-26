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
        
    threshold_min = -10
    threshold_max = 10**1
    
    encoder_inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(32, activation="relu")(encoder_inputs)
    x = layers.Dense(16, activation="relu")(x)
    z = layers.Dense(latent_dim, activation="linear")(x)
    z_log_var = layers.Dense(latent_dim, activation="linear")(x)
    #◘z_log_var = tf.math.maximum(z_log_var, threshold_min)
    #z_log_var = tf.math.minimum(z_log_var, threshold_max)
    encoder = keras.Model(encoder_inputs, [z,z_log_var], name="encoder")
    return encoder

def create_decoder(input_dim,latent_dim):
    
    threshold = -300
    
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(16, activation="relu")(latent_inputs)
    x = layers.Dense(32, activation="relu")(x)
    x_output = layers.Dense(input_dim, activation="linear")(x)
    x_log_var = layers.Dense(input_dim, activation="linear")(x)
    x_log_var = tf.math.maximum(x_log_var, threshold)
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
            
            z,z_log_var = self.encoder(X)
            reconstruction,x_log_var = self.decoder(z)
            
            xx = tf.pow(X - reconstruction,2)
            zz = tf.pow(z_log_var,2)
            xx_log_var = tf.pow(x_log_var,2)
            total_loss = tf.reduce_mean(tf.multiply(tf.reduce_mean(xx,axis=1),y)) + tf.reduce_mean(tf.multiply(tf.reduce_mean(zz,axis=1),y)) #+ tf.reduce_mean(tf.multiply(tf.reduce_mean(xx_log_var,axis=1),y))
            
            
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        
        return {"loss": self.total_loss_tracker.result()}