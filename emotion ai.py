import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd 
import time

FER = pd.read_csv('/Users/alfie/Documents/school /computer science /courswork /FER-2013 data set ')

(x_train, y_train), (x_test, y_test) = FER.load_data()


x_train = tf.keras.utils.normalize(x_train, axis=1 )
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10 )

# model.save('FER_model.h5')

# model = tf.keras.models.load_model('FER_model.h5')


