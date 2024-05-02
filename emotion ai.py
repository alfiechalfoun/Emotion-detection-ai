# imports
import cv2 
import tensorflow.keras as tf
import matplotlib.pyplot as plt 
from tensorflow.keras import datasets, layers, models 
import pandas as pd
import numpy as np

emotions = ['angry','disgust','fear','happy','neutral','sad','surprise']
FER_path = '/Users/alfie/Documents/school /computer science /courswork /FER-2013 data set /fer2013.csv'

#loading the FER_2013 daterset 
faces = pd.read_csv(FER_path)
faces['pixels'] = faces['pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape(48, 48) / 255.0)

# splitting the data into traning and testing 
traning_data = faces[faces['Usage'] == 'Training'] 
test_data =  faces[faces['Usage'] == 'PublicTest']

training_images = np.stack(traning_data['pixels'].values)
training_labels = traning_data['emotion'].values
testing_images = np.stack(test_data['pixels'].values)
testing_labels = test_data['emotion'].values

training_images = training_images.reshape(-1, 48, 48, 1)
testing_images = testing_images.reshape(-1, 48, 48, 1)





# showing the image 
plt.figure(figsize=(10,10))
for i in range (16):
    plt.subplot(4,4,i+1)
    plt.xticks ([])
    plt.yticks ([])
    plt.imshow(testing_images[i,:,:,0], cmap=plt.cm.gray)
    emotions_lable = emotions[testing_labels[i]]
    plt.xlabel(emotions_lable)
plt.show()