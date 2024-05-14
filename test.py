# imports
import cv2 
import tensorflow.keras as tf
import matplotlib.pyplot as plt 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np

# labeling
emotions = ['angry','disgust','fear','happy','neutral','sad','surprise']
FER_path = '/Users/alfie/Documents/school /computer science /courswork /FER-2013 data set /fer2013.csv'

# loading the FER_2013 dataset 
faces = pd.read_csv(FER_path)
faces['pixels'] = faces['pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape(48, 48) / 255.0)

# splitting the data into training and testing 
training_data = faces[faces['Usage'] == 'Training'] 
test_data =  faces[faces['Usage'] == 'PublicTest']

# preparing the data
training_images = np.stack(training_data['pixels'].values)
training_labels = training_data['emotion'].values
testing_images = np.stack(test_data['pixels'].values)
testing_labels = test_data['emotion'].values

# Reshaping images
training_images = training_images.reshape(-1, 48, 48, 1)
testing_images = testing_images.reshape(-1, 48, 48, 1)

# Convert labels to categorical format
num_classes = len(emotions)
training_labels = to_categorical(training_labels, num_classes=num_classes)
testing_labels = to_categorical(testing_labels, num_classes=num_classes)

# Model definition
model = Sequential()
input_shape = (48, 48, 1)
model.add(Conv2D(64, (5, 5), input_shape=input_shape, activation='relu', padding='same'))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Model compilation
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model.summary()


    # # showing the image 
    # plt.figure(figsize=(10,10))
    # for i in range (16):
    #     plt.subplot(4,4,i+1)
    #     plt.xticks ([])
    #     plt.yticks ([])
    #     plt.imshow(testing_images[i,:,:,0], cmap=plt.cm.gray)
    #     emotions_lable = emotions[testing_labels[i]]
    #     plt.xlabel(emotions_lable)
    # plt.show()