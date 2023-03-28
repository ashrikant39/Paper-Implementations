import os
import sys
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
import albumentations as A
from albumentations import  Compose, RandomBrightness, HorizontalFlip, VerticalFlip, Rotate
import math
import numpy as np
import matplotlib.pyplot as plt
import libtiff

def read_tif(path):
  return libtiff.TIFF.open(path).read_image()

def normalize(image):
  return (image-image.min())/(image.max()-image.min())
class CD_DataGenerator(Sequence):

    def __init__(self, csv_file_path, batch_size, image_shape=(512, 512), im_type='rgb', transforms= None):
        super().__init__()
        
        assert im_type in ['rgb', 'multi']
        
        self.df= pd.read_csv(csv_file_path)
        self.batch_size = batch_size
        self.image_shape= image_shape
        self.transforms= transforms
        self.image_type= im_type
        
    def __len__(self):
      return math.floor(len(self.df) / self.batch_size)

    def load_single_set(self, index):
      
      if self.image_type is 'rgb':
        first_image= cv2.resize(cv2.imread(self.df.iloc[index]['A'])[:,:,::-1], self.image_shape, interpolation=cv2.INTER_CUBIC)
        second_image= cv2.resize(cv2.imread(self.df.iloc[index]['B'])[:,:,::-1], self.image_shape, interpolation=cv2.INTER_CUBIC)
      
      elif self.image_type is 'multi':
        first_image= read_tif(self.df.iloc[index]['A'])
        second_image= read_tif(self.df.iloc[index]['B'])
        
      _, label= cv2.threshold(cv2.imread(self.df.iloc[index]['label'], 0), 128, 255, cv2.THRESH_BINARY)
      label= cv2.resize(label, self.image_shape, interpolation=cv2.INTER_NEAREST)
      if self.transforms is not None:
        aug_data = self.transforms(image=first_image, image1=second_image, mask=label)
        first_image, second_image, label = aug_data["image"], aug_data["image1"], aug_data["mask"]
      
      return normalize(first_image), normalize(second_image), label/255.

    def __getitem__(self, idx):
      
      if self.image_type is 'rgb':
        batch_first= np.zeros((self.batch_size, *self.image_shape, 3))
      
      elif self.image_type is 'multi':
        batch_first= np.zeros((self.batch_size, *self.image_shape, 13))
          
      batch_second= np.zeros_like(batch_first)
      label_batch= np.zeros((self.batch_size, *self.image_shape, 1))

      for i in range(self.batch_size):
        batch_first[i], batch_second[i], label_batch[i,:,:,0]= self.load_single_set(idx+i)
      
      return np.concatenate((batch_first, batch_second), axis=-1), label_batch
      
        
def get_data_generators(train_csv_path, test_csv_path, val_csv_path, image_shape, batch_size, im_type='rgb', transforms=None):
    train_datagen = CD_DataGenerator(train_csv_path, batch_size, image_shape, im_type=im_type, transforms=transforms)
    test_datagen= CD_DataGenerator(test_csv_path, batch_size, image_shape, im_type=im_type)
    val_datagen= CD_DataGenerator(val_csv_path, batch_size, image_shape, im_type=im_type)
    
    return train_datagen, test_datagen, val_datagen

def plotter(datagen, concat_flag=False):
    images, labels= next(iter(datagen))
    for i in range(datagen.batch_size):
      
      fig= plt.figure(figsize=(20, 5))
      ax1= fig.add_subplot(1,3,1)
      ax1.set_title('First Image')
      ax1.imshow(images[i,:,:,:3])
      
      ax2= fig.add_subplot(1,3,2)
      ax2.set_title('Second Image')
      ax2.imshow(images[i,:,:,3:])  
    
      ax3= fig.add_subplot(1,3,3)
      ax3.set_title('Change Mask')
      ax3.imshow(labels[i,:,:,0], 'gray')  
    
      plt.show()    
