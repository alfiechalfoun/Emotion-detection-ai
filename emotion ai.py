import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd 
import time

FER = pd.read_csv('/Users/alfie/Documents/school /computer science /courswork /FER-2013 data set ')




# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis=1 )
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train,y_train,epochs=10 )

# model.save('handwriting_model.h5')

# model = tf.keras.models.load_model('handwriting_model.h5')


# loss, accuracy = model.evaluate(x_test, y_test)
# print(loss)
# print(accuracy)

# image_number = 1

# while os.path.isfile(f'/Users/alfie/Documents/school /computer science /courswork /number ai /MINIS data set/testSet/testSet/img_{image_number}.jpg'):
#     try:
#         image_path = f'/Users/alfie/Documents/school /computer science /courswork /number ai /MINIS data set/testSet/testSet/img_{image_number}.jpg'
#         # Load the image
#         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         image = np.array([image])
#         prediction = model.predict(image)
#         print(f'the number is probably a {np.argmax(prediction)}')
#         plt.imshow(image[0], cmap=plt.cm.binary)
#         plt.show()
#         time.sleep(1)
#     except Exception as e:
#         print(f'error is {e}')
#     finally:
#         image_number += 1