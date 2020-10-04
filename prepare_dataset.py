# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 09:27:25 2020

@author: radae
"""

# File manipulation libraries.
import os
import glob
import shutil

# Prepares a new folder structure for pictures.
curr_dir = os.getcwd()
print("Current directory: ",curr_dir)

# Goes to training set folder.

from_dir = 'Training'
os.chdir(from_dir)
from_dir=os.getcwd()

print("Current directory: ",from_dir)

# Gets a list of folders in the directory.
folders = glob.glob('*')

list_names=[]

selected_list = ["Apple","Banana","Cherry","Grape","Plum","Pepper","Tomato","Potato","Pear","Peach"]

destination_directory = 'New_training'
os.chdir('..')
os.chdir('New_training')
destination_directory = os.getcwd()


# For all folders, takes the first word of the name.
for folder in folders:
    folder_original_name = folder
    folder_list_name = folder_original_name.split('\\')
    folder_len_list = len(folder_list_name)
    folder_new_name = folder_list_name[folder_len_list-1]
    folder_new_name = (folder_new_name.split())[0]
    print(folder_new_name)
    
    # Verifies if the fruit type should be considered.
    print("Considering: ",folder_new_name)
    if folder_new_name in selected_list:
        # Verifies if there exists a folder in the destination with the same name.
        print("Now checking for name ", folder_new_name,  " in list ", list_names)
        if folder_new_name in list_names:
            # In this case the folder already exists.
            # Changes the focus to the destination directory.
            os.chdir(destination_directory)
            os.chdir(folder_new_name)
            # Gets the number of the last element in the folder.
            number_files_in_destination = len(glob.glob('*'))
            print("There are already ",number_files_in_destination, " files in the ",folder_new_name," directory.")
            # Copies the files.
            # Changes the focus to the from directory.
            os.chdir(from_dir + "\\" + folder_original_name)
            list_from_files = glob.glob('*')
            counter=0
            for file in list_from_files:
                source_path = from_dir +  "\\" + folder_original_name + "\\" + file
                new_number=number_files_in_destination + counter
                counter = counter+1
                dest_path = destination_directory  + "\\" + folder_new_name + "\\" + str(new_number) + ".jpg"
                shutil.copy(source_path,dest_path)
            print("Successfully copied from ",folder_original_name," to ", folder_new_name)
        else:
            # In this case the folder does not exist.
            # Changes the focus to the destination directory.
            os.chdir(destination_directory)
            print('Current working directory: ',os.getcwd())
            # Creates the new folder.
            os.mkdir(folder_new_name)
            print("Created folder ",folder_new_name)
            # Copies the file
            os.chdir(from_dir+'\\'+ folder_original_name)
            print('Current working directory: ',os.getcwd())
            list_from_files = glob.glob('*')
            counter = 0
            for file in list_from_files:
                source_path = from_dir +  "\\" + folder_original_name + "\\" + file
                new_number = counter
                counter = counter+1
                dest_path = destination_directory  + "\\" + folder_new_name + "\\" + str(new_number) + ".jpg"
                shutil.copy(source_path,dest_path)
            print("Successfully copied from ",folder_original_name," to ", folder_new_name)
            list_names.append(folder_new_name)
    else:
        print("Fruit type ",folder_new_name," discarded")

print("Successfully prepared training dataset")

curr_dir = os.getcwd()
print("Current directory: ",curr_dir)

os.chdir(from_dir)
os.chdir('..')
from_dir = 'Test'
os.chdir(from_dir)
from_dir=os.getcwd()

print("Current directory: ",from_dir)

# Gets a list of folders in the directory.
folders = glob.glob('*')

list_names=[]

selected_list = ["Apple","Banana","Cherry","Grape","Plum","Pepper","Tomato","Potato","Pear","Peach"]

destination_directory = 'New_test'
os.chdir('..')
os.chdir(destination_directory)
destination_directory=os.getcwd()

# For all folders, takes the first word of the name.
for folder in folders:
    folder_original_name = folder
    folder_list_name = folder_original_name.split('\\')
    folder_len_list = len(folder_list_name)
    folder_new_name = folder_list_name[folder_len_list-1]
    folder_new_name = (folder_new_name.split())[0]
    print(folder_new_name)
    
    # Verifies if the fruit type should be considered.
    print("Considering: ",folder_new_name)
    if folder_new_name in selected_list:
        # Verifies if there exists a folder in the destination with the same name.
        print("Now checking for name ", folder_new_name,  " in list ", list_names)
        if folder_new_name in list_names:
            # In this case the folder already exists.
            # Changes the focus to the destination directory.
            os.chdir(destination_directory)
            os.chdir(folder_new_name)
            # Gets the number of the last element in the folder.
            number_files_in_destination = len(glob.glob('*'))
            print("There are already ",number_files_in_destination, " files in the ",folder_new_name," directory.")
            # Copies the files.
            # Changes the focus to the from directory.
            os.chdir(from_dir + "\\" + folder_original_name)
            list_from_files = glob.glob('*')
            counter=0
            for file in list_from_files:
                source_path = from_dir +  "\\" + folder_original_name + "\\" + file
                new_number=number_files_in_destination + counter
                counter = counter+1
                dest_path = destination_directory  + "\\" + folder_new_name + "\\" + str(new_number) + ".jpg"
                shutil.copy(source_path,dest_path)
            print("Successfully copied from ",folder_original_name," to ", folder_new_name)
        else:
            # In this case the folder does not exist.
            # Changes the focus to the destination directory.
            os.chdir(destination_directory)
            # Creates the new folder.
            os.mkdir(folder_new_name)
            print("Created folder ",folder_new_name)
            # Copies the files.
            os.chdir(from_dir + "\\" + folder_original_name)
            list_from_files = glob.glob('*')
            counter = 0
            for file in list_from_files:
                source_path = from_dir +  "\\" + folder_original_name + "\\" + file
                new_number = counter
                counter = counter+1
                dest_path = destination_directory  + "\\" + folder_new_name + "\\" + str(new_number) + ".jpg"
                shutil.copy(source_path,dest_path)
            print("Successfully copied from ",folder_original_name," to ", folder_new_name)
            list_names.append(folder_new_name)
    else:
        print("Fruit type ",folder_new_name," discarded")

print("Successfully prepared test dataset")
        