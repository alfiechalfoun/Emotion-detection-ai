import tensorflow as tf
from sklearn.metrics import confusion_matrix
from emotion_ai import Modle
from sklearn.utils.multiclass import unique_labels
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Matrix(Modle):
    def __init__(self):
        super().__init__()
        super().prepering_data()
        super().load_modle()
        self.labels = []
        self.predictions = []

    def get_prediction(self): #len(self.testing_images)
        for i in range(len(self.testing_images)):
             self.labels.append(self.emotions[self.testing_labels[i]])
             self.predictions.append(super().predict(self.testing_images[i]))
            #  print(self.labels, self.predictions)

    def create_matrix(self):
        self.get_prediction()
        self.matrix = confusion_matrix(self.labels, self.predictions)
        self.actual_lable = [f'inp {emotion}' for emotion in self.emotions]
        self.prediction_lables = [f'pred {emotion}' for emotion in self.emotions]
        self.table = pd.DataFrame(self.matrix, index= unique_labels(self.actual_lable), columns = unique_labels(self.prediction_lables))    
        
    
    def display_matrx(self):
        self.create_matrix()
        print(self.matrix)
        
    def dispaly_table (self):
        self.create_matrix()
        print(self.table)

    def display_heatmap(self):
        self.create_matrix()
        sns.heatmap(self.table, annot= True, fmt='d', cmap='plasma', vmax= 400)
        plt.show()

matrix = Matrix()
matrix.display_heatmap()