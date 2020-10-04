# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:12:35 2020

@author: radae
"""

import matplotlib.pyplot as plt
import PIL
import PIL.Image
import numpy as np
import pathlib
import os
import glob
import random

import tensorflow as tf

from tensorflow.keras import layers

def learn():
    # Basic ML setup.
    batch_size = 32
    img_height = 32
    img_width = 32
    
    epoch_number = 5
    
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
        
    return probability_model,class_names

def load_random_image():
    # Chooses a random image from the test set.
    # Moves to the right path.
    print('Current working directory at the beginning of load_random_image: ',os.getcwd())
    from_path='New_test'
    os.chdir(from_path)
    from_path=os.getcwd()
    list_types = glob.glob('*')
    number_types = len(list_types)
    
    # Chooses a random type.
    index = int(random.uniform(0,number_types))
    
    type_fruit=list_types[index]
    
    # Moves to the selected directory.
    os.chdir(type_fruit)
    
    list_images = glob.glob('*')
    number_images = len(list_images)
    
    which_file = int(random.uniform(0,number_images))
    
    name_file = from_path + "\\" + type_fruit + "\\" + str(which_file) + ".jpg"
    
    return name_file

def prepare_image(path,image_size):
    # Prepares an image for the prediction.
    img=tf.keras.preprocessing.image.load_img(path,target_size=image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    return img_array

def predict(path,image_size,probability_model,class_names):
    # Returns the prediction for an image.
    immy=prepare_image(path,image_size)
    predictions = probability_model.predict(immy)
    type_index=np.argmax(predictions)
    return class_names[np.argmax(predictions)]

def get_number_for_category(category,class_names):
    # Returns the numeric code for category.
    
    if category in class_names:
        return class_names.index(category)
    else:
        return -1
    
    
def get_picture(matrix):
    new_matrix_size = 1000
    old_matrix_size = len(matrix[0])
    cell_size = new_matrix_size /  old_matrix_size
    
    new_matrix = np.zeros((new_matrix_size,new_matrix_size))
    
    matrix = matrix.astype(int)
    
    # Cycle on the rows.
    for old_row in range(0,old_matrix_size):
        for old_col in range(0,old_matrix_size):
            for j in range(int(old_row*cell_size),int(old_row*cell_size + cell_size)):
                for k in range(int(old_col*cell_size),int(old_col*cell_size + cell_size)):
                    new_matrix[j][k] = matrix[old_row][old_col]
    
    # Gets the picture.
    zeros_matrix = np.zeros((new_matrix_size,new_matrix_size))
    
    new_matrix = new_matrix.astype(int)
    
    img = PIL.Image.fromarray(np.uint8(new_matrix),'L')
   
    return img,new_matrix

# This code looks for errors in prediction selecting random images.
# First, the network should be trained. The training returns a probability model.
print('Now beginning learning procedures')
pm,class_names = learn()
print('Working directory after learning: ',os.getcwd())

number_tempt = 500

error_list = []
wrong_pred_list=[]

for counter in range(0,number_tempt):
    # Selects a random image.
    perc = load_random_image()
    os.chdir('..')
    os.chdir('..')
    #print('Random image path: ',perc)    
    
    # Gets the right label.
    list_perc = perc.split('\\')
    right_fruit_type=list_perc[len(list_perc)-2]
    
    image_size=(32,32)
    
    predicted_fruit_type = predict(perc,image_size,pm,class_names)
    if (predicted_fruit_type != right_fruit_type):
        error_list.append(perc)
        wrong_pred_list.append((predicted_fruit_type,right_fruit_type))
        print("Predicted: ", predicted_fruit_type, "  Right: ",right_fruit_type)
        
    print("Checked image ",counter,"/",number_tempt," ",int(counter/number_tempt*100),"%")
    
    # Creates a all-zero matrix.
M = np.zeros((10,10))

for element in wrong_pred_list:
    predicted_type = element[0]
    right_type = element[1]
    
    predicted_number = get_number_for_category(predicted_type,class_names)
    right_number = get_number_for_category(right_type,class_names)
    
    # Increases by one the value of the element in the matrix.
    M[predicted_number][right_number] = M[predicted_number][right_number] +1
    
# Rescalate with the max of the matrix.

maximum_error = np.amax(M)
M = M/maximum_error * 255.

M = M.astype(int)

print(M)

img,new_matrix = get_picture(M)
img.show()