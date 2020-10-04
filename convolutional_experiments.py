# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 07:59:55 2020

@author: radae
"""

import numpy as np
import sys
import os
import PIL
import PIL.Image
import tensorflow as tf

import pathlib

import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

def train_neural_network(hyper_pars):
    # Trains the NN using the given hyperparameters.
    # Basic ML setup.
    batch_size = 32
    img_height = 32
    img_width = 32
    
    epoch_number = 15   
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
    
    train_data_directory = 'New_training'
    train_data_directory = pathlib.Path(train_data_directory)
    
    test_data_directory = 'New_test'
    test_data_directory = pathlib.Path(test_data_directory)
    
    # Loads the training set.
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      train_data_directory,
      seed=123,
      image_size=(img_height, img_width),
      batch_size=hyper_pars['batch_size'])
    
    print("Successfully loaded training set.")
    
    # Loads the test set.
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      test_data_directory,
      seed=123,
      image_size=(img_height, img_width),
      batch_size=hyper_pars['batch_size'])
    
    print("Successfully loaded test set.")
    
    class_names = train_ds.class_names
    
    # Performance optimization.
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    num_classes = 10
    
    # Here the NN code.
    model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(hyper_pars['output_space_dimension'], hyper_pars['size_conv_window'], activation=hyper_pars['activation_function']),
    layers.Flatten(),
    layers.Dense(num_classes)])
    
    # Here the possibility to load previous weights.
    load_previous = False
    if load_previous:
        model.load_weights('./checkpoints/my_checkpoint')
    
    # Model compilation. Here losses, metrics and optimizers.
    opt = tf.keras.optimizers.SGD(learning_rate=hyper_pars['learning_rate'])
    
    model.compile(
    optimizer=opt,
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
    
    
    # Model fitting. Here epochs.
    history=model.fit(
    train_ds,
    batch_size=batch_size,
    validation_data=val_ds,
    epochs=epoch_number,
    callbacks=[callback],
    verbose=1
    )
    
    model.summary()
    
    return history


# From here the main code.
# Defines the dictionary with NN hyperparameters.
dict_hp={'output_space_dimension':32,
         'size_conv_window':3,
         'max_pooling_size':2,
         'learning_rate':0.06,
         'batch_size':32,
         'activation_function':'relu'
         }    

# Lists for the results.
arr_activations = []
arr_training_errors = []
arr_validation_errors = []
                                             
# Trains the network with the given hyperparameters, cycling on possible values for the batch size.
possible_activations = ['relu','tanh','selu','sigmoid']
for activation in possible_activations:
    dict_hp['activation_function'] = activation
    history = train_neural_network(dict_hp)

    # Last values for training and validation errors.
    # arr_len = len(history.history['accuracy'])
    # training_error = (history.history['accuracy'])[arr_len-1]
    # last_validation_error = (history.history['val_accuracy'])[arr_len-1] 
    
    # In the case of activation functions I'm interested in the whole training process.
    training_error=history.history['accuracy']
    validation_error=history.history['val_accuracy']
    
    # Saves the results in the lists.
    arr_activations.append(activation)
    arr_training_errors.append(training_error)
    arr_validation_errors.append(validation_error)
    
    print('Ok for activation function = ', activation)