import tensorflow as tf 
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from emotion_ai import Modle, FER13Data

class Imiges(Modle):
    def __init__(self):
        super().__init__()
        super().load_modle()
        super().prepering_data()
    
    def display_imige(self):
        for i in range(100):
            image = self.testing_images[i]
            prediction = super().predict(image)
            self.emotion_label = self.emotions[self.testing_labels[i]]
            plt.imshow(image, cmap=plt.cm.gray)
            plt.xlabel(f'the prediction is {prediction}')
            plt.ylabel(f'the emotion is {self.emotion_label}')
            plt.show()

if __name__ == '__main__':
    imig = Imiges()
    imig.display_imige()