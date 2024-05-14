import tensorflow as tf 
import matplotlib as plt 
from tensorflow.keras.models import load_model
from emotion_ai import loading_data
import pandas as pd 
model_path = ('emotion_detection_model/emotion_detection_modle.h5')

model = load_model(model_path)

training_images, training_labels, testing_images, testing_labels = loading_data()

test_loss, test_accuracy = model.evaluate(testing_images, testing_labels)
print(f'the loss is {test_loss} \n the acuracy is {test_accuracy}')