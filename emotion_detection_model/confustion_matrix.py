import tensorflow as tf
from sklearn.metrics import confusion_matrix
from emotion_ai import Modle
from sklearn.utils.multiclass import unique_labels
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# creat a confustion matrix of the model accuracy 
class Matrix(Modle):
    def __init__(self):
        super().__init__()
        super().prepering_data()
        super().load_modle()
        self.labels = []
        self.predictions = []

    def get_prediction(self): 
        for i in range(len(self.testing_images)):
            self.labels.append(self.emotions[self.testing_labels[i]])  # ✅ Actual emotion label

            # ✅ Extract only the predicted label from predict()
            predicted_label, _ = super().predict(self.testing_images[i])  # Ignore confidence
            self.predictions.append(predicted_label)  # ✅ Store only the label


    # uses the prediction and label to creat a confution matric 
    def create_matrix(self):
        self.get_prediction()
        self.matrix = confusion_matrix(self.labels, self.predictions)
        self.actual_lable = [f'inp {emotion}' for emotion in self.emotions]
        self.prediction_lables = [f'pred {emotion}' for emotion in self.emotions]
        self.table = pd.DataFrame(self.matrix, index= unique_labels(self.actual_lable), columns = unique_labels(self.prediction_lables))   

    # normilise the marix(turn into pecentage)
    def normalize_matrix(self):
        self.create_matrix()
        self.matrix_normalized = self.matrix.astype('float') / self.matrix.sum(axis=1)[:, np.newaxis] * 100
        self.table_normalized = pd.DataFrame(self.matrix_normalized, index=unique_labels(self.actual_lable), columns=unique_labels(self.prediction_lables)) 
        
    # display the confution matrix
    def display_matrx(self):
        self.create_matrix()
        print(self.matrix)
        
    def dispaly_table (self):
        self.create_matrix()
        print(self.table)

    # turn matrix into heat map
    def display_heatmap(self):
        self.create_matrix()
        sns.heatmap(self.table, annot= True, fmt='d', cmap='plasma', vmax= 400)
        plt.show()

    # turn matrix into normilzed heat map
    def display_normalised_heatmap(self):
        self.normalize_matrix()
        sns.heatmap(self.table_normalized, annot= True, fmt='.2f', cmap='plasma', vmax= 100)
        plt.show()

if __name__ == '__main__':
    matrix = Matrix()
    matrix.display_normalised_heatmap()
    