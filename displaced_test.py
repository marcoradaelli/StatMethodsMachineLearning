# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:09:43 2020

@author: radae
"""
import numpy as np
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
    
    train_data_directory = 'Disl_training'
    train_data_directory = pathlib.Path(train_data_directory)
    
    test_data_directory = 'Disl_test'
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
    layers.Conv2D(hyper_pars['output_space_dimension'], hyper_pars['size_conv_window'], activation='relu'),
    # layers.MaxPooling2D(),
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
    
  #  model.summary()
    
    return history


# From here the main code.
# Defines the dictionary with NN hyperparameters.
dict_hp={'output_space_dimension':32,
         'size_conv_window':3,
         'max_pooling_size':2,
         'learning_rate':0.06,
         'batch_size':32
         }    
 
# Trains the NN.                                            
history = train_neural_network(dict_hp)

# Retries the learning process history.
arr_training = np.array(history.history['accuracy'])
arr_validation = np.array(history.history['val_accuracy'])
x = np.arange(0,14)    # Till the last epoch number