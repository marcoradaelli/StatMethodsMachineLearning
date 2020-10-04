# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 16:23:24 2020

@author: radae

The purpose of this program is to challenge the translational invariance of the convolutional NN. 
It loads the pictures from both validation and training datasets (100x100 px) and creates new pictures (150x150 px) with dislocated images by a random quantity.
"""

import random
from PIL import Image
import os
import glob
import shutil

def dislocate_image(old_img):
    # Chooses at random a dislocation vector in [0,49]
    x_disl = random.randint(0,49)
    y_disl = random.randint(0,49)
    
    # Creates a new empty image.
    new_img = Image.new('RGB', (150, 150), color = 'white')
    
    # Loads pixel by pixel.
    old_pixels = old_img.load()
    new_pixels = new_img.load()
    
    # Copies pixel by pixel with the right translation.
    for x in range(0,100):
        for y in range(0,100):
            new_pixels[x+x_disl,y+y_disl] = old_pixels[x,y]
    
    # Returns the new image.
    return new_img
    
# Here the main code.
# From and to dataset position.
from_dir = 'New_training'
to_dir = 'Disl_training'
os.mkdir(to_dir)
os.chdir(to_dir)
to_dir=os.getcwd()
print('Current working directory: ',os.getcwd())
os.chdir('..')
os.chdir(from_dir)
from_dir=os.getcwd()

# Cycles over the folders.
folders = glob.glob('*')
for folder in folders:
    # Creates a new folder in the destination path with the same name.
    os.mkdir(to_dir + '\\' + folder)
    
    # Cycles over the files in the folder.
    os.chdir(from_dir)
    os.chdir(folder)
    files = glob.glob('*')
    for file in files:
        # Loads image.
        img = Image.open(file)
        # Dislocates the image.
        new_img = dislocate_image(img)
        
        # Creates the name of the new file.
        file_out_name = to_dir + '\\' + folder + '\\' + file
        
        # Saves the new image.
        new_img.save(file_out_name)
        
        print('Successfully dislocated image ', file)
        
    print('Completed dislocation for training folder: ', folder)

os.chdir('..')
os.chdir('..')

# From and to dataset position.
from_dir = 'New_test'
to_dir = 'Disl_test'
os.mkdir(to_dir)

os.chdir(from_dir)
from_dir=os.getcwd()
os.chdir('..')
os.chdir(to_dir)
to_dir=os.getcwd()
os.chdir(from_dir)

# Cycles over the folders.
folders = glob.glob('*')
for folder in folders:
    # Creates a new folder in the destination path with the same name.
    os.chdir('..')
    print('Current working directory (1): ',os.getcwd())
    os.mkdir(to_dir + '\\' + folder)
    print('Folder created')
    
    # Cycles over the files in the folder.
    os.chdir(from_dir)
    os.chdir(folder)
    files = glob.glob('*')
    for file in files:
        # Loads image.
        img = Image.open(file)
        # Dislocates the image.
        new_img = dislocate_image(img)
        
        # Creates the name of the new file.
        file_out_name = to_dir + '\\' + folder + '\\' + file
        
        # Saves the new image.
        new_img.save(file_out_name)
        
        print('Successfully dislocated image ', file)
        
    print('Completed dislocation for test folder: ', folder)