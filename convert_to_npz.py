#author GEC_Batch_23
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#insert ur path here ( should be in directory where train and test folder is there)
input_path = 'C:/Users/T.S. Vishwak/Desktop/dummy/'

def process_data(img_dims,t):

    data = []
    labels = []

    for cond in ['/normal/', '/pneumonia/']:
        for img in (os.listdir(input_path + t + cond)):
           # using .covert() for grayscale
            img = Image.open(input_path+ t +cond+img).convert('L')
           #resize
            img=img.resize((img_dims, img_dims),)
          #normalization
            img = img.astype('float32') / 255
            if cond=='/normal/':
                label = 0
            elif cond=='/pneumonia/':
                label = 1
            data.append(img)
            labels.append(label)
    data=np.array(data)
    labels = np.array(labels)
    
    return  test_data, test_labels

img_dims = 64

train_x, train_y= process_data(img_dims,"train")
test_x, test_y = process_data(img_dims,"test")

print("train labels")
print(train_y)
print("test labels")
print(test_y)
print(test_x.shape)
#test_x=np.reshape(test_x,(-1,28,28,1))
#train_x=np.reshape(train_x,(-1,28,28,1))
print(train_x.shape)
print(test_x.shape)

# save to output path
#np.savez_compressed('C:/Users/T.S. Vishwak/Desktop/dummy/dataset.npz',x_train=train_x,y_train=train_y,x_test=test_x,y_test=test_y)
