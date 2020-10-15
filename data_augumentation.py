#!/usr/bin/env python
# coding: utf-8
##############coded by GEC_BATCH_23##########################
#juypter-notebook




# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse
import os
from PIL import Image,ImageOps
print("modules imported")


#set input path
input_path="F:/pneumonia dataset - Copy/chest_xray/"


def process_data(t, cond,k):
        for img in (os.listdir(input_path + t + cond)):
            image = load_img(input_path + t + cond+img)
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

	    #augumentation
            aug=ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
                zoom_range = 0.2, # Randomly zoom image 
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip = True,  # randomly flip images
                vertical_flip=False)
            total = 0
            imageGen = aug.flow(image, batch_size=1, save_to_dir=(input_path+t+cond),
                save_prefix="image", save_format="jpg")

            # loop over examples from our image data augmentation generator
            for image in imageGen:
                # increment our counter
                total += 1
                # if we have reached the specified number of examples, break
                # from the loop
                if total == k:
                    break
             

#test train *3
#Third parameter is the value k 
# k stands for number of extra agumented image created from one original image 
# so if k is 2 it means from one image we get 2 new agumented images .
#process_data("test","/NORMAL/",3)
#process_data("val","/NORMAL/",1)                    
process_data("train","/NORMAL/",2)                    

