import matplotlib.pyplot as plt

class ploting_training(Modle):
    def __init__(self):
        super().__init__()
        
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