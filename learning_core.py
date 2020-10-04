# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 10:53:35 2020

@author: radae
"""

# This script can be both used alone or called via the learn function, returning the probability model.

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf

import pathlib

import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import regularizers

def learn():
    # Basic ML setup.
    batch_size = 32
    img_height = 32
    img_width = 32
    
    epoch_number = 15
    
    train_data_directory = 'New_training'
    
    train_data_directory = pathlib.Path(train_data_directory)
    
    test_data_directory = 'New_test'
    
    test_data_directory = pathlib.Path(test_data_directory)
    
    # Loads the training set.
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      train_data_directory,
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
    
    print("Successfully loaded training set.")
    
    # Loads the test set.
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
      test_data_directory,
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)
    
    print("Successfully loaded test set.")
    
    class_names = train_ds.class_names
    print(class_names)
    
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
        
    # Rescaling layer (to pass from 0-255 RGB values to 0-1 values.)
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    
    # Performance optimization.
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    num_classes = 10
    
    # Here the NN code.
    model = tf.keras.Sequential([
      #  Here the CNN suggested by Keras tutorial. /Arch. 1
        # layers.experimental.preprocessing.Rescaling(1./255),
        # layers.Conv2D(32, 3, activation='relu'),
        # layers.MaxPooling2D(),
        # layers.Conv2D(32, 3, activation='relu'),
        # layers.MaxPooling2D(),
        # layers.Conv2D(32, 3, activation='relu'),
        # layers.MaxPooling2D(),
        # layers.Flatten(),
        # layers.Dense(128, activation='relu'),
        # layers.Dense(num_classes)
      
      # Arch. 3
        layers.experimental.preprocessing.Rescaling(1./255),
        layers.Flatten(),
        layers.Dense(10,activation='selu'),
        layers.Dense(10,activation='selu'),
        layers.Dense(10,activation='selu'),
        layers.Dense(num_classes)
        
        # Arch. 4
        # layers.experimental.preprocessing.Rescaling(1./255),
        # layers.Flatten(),
        # layers.Dense(10,activation='relu'),
        # layers.Dense(10,activation='relu'),
        # layers.Dense(num_classes)
        
        # Arch. 5
        # layers.experimental.preprocessing.Rescaling(1./255),
        # layers.Flatten(),
        # layers.Dense(10,activation='relu'),
        # layers.Dense(10,activation='elu'),
        # layers.Dense(10,activation='relu'),
        # layers.Dense(5,activation='elu'),
        # layers.Dense(num_classes)
        
        # Arch. 6
        # layers.experimental.preprocessing.Rescaling(1./255),
        # layers.Conv2D(32, 3, activation='relu'),
        # layers.Flatten(),
        # layers.Dense(128, activation='relu'),
        # layers.Dense(num_classes)
        
         # Arch. 7
        # layers.experimental.preprocessing.Rescaling(1./255),
        # layers.Conv2D(32, 3, activation='relu'),
        # layers.Flatten(),
        # layers.Dense(num_classes)
    ])
    
    # Here the possibility to load previous weights.
    load_previous = False
    if load_previous:
        model.load_weights('./checkpoints/my_checkpoint')
    
    # Model compilation. Here losses, metrics and optimizers.
    model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
    
    
    # Model fitting. Here epochs.
    history=model.fit(
    train_ds,
    batch_size=batch_size,
    validation_data=val_ds,
    epochs=epoch_number,
    verbose=1
    )
    
    model.summary()
    
    # Saving the model.
    model.save_weights('./checkpoints/my_checkpoint')
    
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    
    plotting_utility=True
    if plotting_utility:
        train_acc = np.array(history.history['accuracy'])
        val_acc = np.array(history.history['val_accuracy'])
        x = np.arange(0,epoch_number,1)
        plt.title("Accuracy")
        plt.plot(x,train_acc,val_acc)
        plt.show()
        
    return probability_model
    
# Basic ML setup.
batch_size = 32
img_height = 32
img_width = 32

epoch_number = 15

train_data_directory = 'New_training'

train_data_directory = pathlib.Path(train_data_directory)

test_data_directory = 'New_test'

test_data_directory = pathlib.Path(test_data_directory)

# Loads the training set.
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_data_directory,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

print("Successfully loaded training set.")

# Loads the test set.
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_data_directory,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

print("Successfully loaded test set.")

class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
# Rescaling layer (to pass from 0-255 RGB values to 0-1 values.)
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

# Performance optimization.
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 10

# Here the NN code.
model = tf.keras.Sequential([
  #  Here the CNN suggested by Keras tutorial. /Arch. 1
    # layers.experimental.preprocessing.Rescaling(1./255),
    # layers.Conv2D(32, 3, activation='relu'),
    # layers.MaxPooling2D(),
    # layers.Conv2D(32, 3, activation='relu'),
    # layers.MaxPooling2D(),
    # layers.Conv2D(32, 3, activation='relu'),
    # layers.MaxPooling2D(),
    # layers.Flatten(),
    # layers.Dense(128, activation='relu'),
    # layers.Dense(num_classes)
  
  # Arch. 3
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Flatten(),
    layers.Dense(10,activation='selu'),
    layers.Dense(10,activation='selu'),
    layers.Dense(10,activation='selu'),
    layers.Dense(num_classes)
    
    # Arch. 4
    # layers.experimental.preprocessing.Rescaling(1./255),
    # layers.Flatten(),
    # layers.Dense(10,activation='relu'),
    # layers.Dense(10,activation='relu'),
    # layers.Dense(num_classes)
    
    # Arch. 5
    # layers.experimental.preprocessing.Rescaling(1./255),
    # layers.Flatten(),
    # layers.Dense(10,activation='relu'),
    # layers.Dense(10,activation='elu'),
    # layers.Dense(10,activation='relu'),
    # layers.Dense(5,activation='elu'),
    # layers.Dense(num_classes)
    
    # Arch. 6
    # layers.experimental.preprocessing.Rescaling(1./255),
    # layers.Conv2D(32, 3, activation='relu'),
    # layers.Flatten(),
    # layers.Dense(128, activation='relu'),
    # layers.Dense(num_classes)
    
     # Arch. 7
    # layers.experimental.preprocessing.Rescaling(1./255),
    # layers.Conv2D(32, 3, activation='relu'),
    # layers.Flatten(),
    # layers.Dense(num_classes)
])

# Here the possibility to load previous weights.
load_previous = False
if load_previous:
    model.load_weights('./checkpoints/my_checkpoint')

# Model compilation. Here losses, metrics and optimizers.
model.compile(
optimizer='adam',
loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])


# Model fitting. Here epochs.
history=model.fit(
train_ds,
batch_size=batch_size,
validation_data=val_ds,
epochs=epoch_number,
verbose=1
)

model.summary()

# Saving the model.
model.save_weights('./checkpoints/my_checkpoint')

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

plotting_utility=True
if plotting_utility:
    train_acc = np.array(history.history['accuracy'])
    val_acc = np.array(history.history['val_accuracy'])
    x = np.arange(0,epoch_number,1)
    plt.title("Accuracy")
    plt.plot(x,train_acc,val_acc)
    plt.show()