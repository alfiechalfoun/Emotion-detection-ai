import matplotlib.pyplot as plt
from emotion_ai import modle

# plot a graph of acuracy aggents epocs 
class ploting_training(Modle):
    def __init__(self):
        super().__init__()
        
    # trains the model and then take the test acuracy agent training accuracy 
    # (plot accuracy of imiges the model is traind on and accuracy of imiges the model had never seen)
    def plot_accuracy(self):
        super().train_modle()
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['accuracy'], marker='o', label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], marker='o', label='Validation Accuracy')
        plt.title('Model Accuracy vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    graph = ploting_training()
    graph.plot_accuracy()