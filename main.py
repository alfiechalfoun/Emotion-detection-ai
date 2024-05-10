import pydot
from keras.utils import plot_model
import tensorflow as tf

model = tf.keras.models.load_model('emotion_detection_modle.h5')

plot_model(model, to_file='/Users/alfie/Documents/school /computer science ')
