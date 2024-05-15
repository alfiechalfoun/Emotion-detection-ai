import tensorflow as tf 
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from emotion_ai import Modle, FER13Data

# data = FER13Data()

modle = Modle()
modle.get_modle_layout()
