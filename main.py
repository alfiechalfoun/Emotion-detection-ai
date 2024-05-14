import pydot
from keras.utils import plot_model
import tensorflow as tf
from tensorflow.keras.models import load_model


model = load_model('emotion detection model/emotion_detection_modle.h5')

plot_model(
    model, 
    to_file='/Users/alfie/Documents/school /computer science ',
    show_shapes= True,
    show_layer_names=True
    )

print('sucsesfull')