# 60 pecent acuracy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D,BatchNormalization  # type: ignore
from tensorflow.keras.metrics import categorical_accuracy  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.optimizers import *  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore

# class for loading and prepering the FER_2013 daterset
class FER13Data():
    def __init__(self):
        # lableing the dater 
        self.emotions = ['angry','disgust','fear','happy','sad','surprise','neutral']
        self.FER_path = '/Users/alfie/Documents/school /computer science /courswork /FER-2013 data set /fer2013.csv' # change to where the file is

        #loading the FER_2013 daterset 
        self.faces = pd.read_csv(self.FER_path)
        self.faces['pixels'] = self.faces['pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape(48, 48) / 255.0)

        self.traning_data = self.faces[self.faces['Usage'] == 'Training'] 
        self.test_data =  self.faces[self.faces['Usage'] == 'PublicTest']

    # Prepare and reshape the image data and labels for training and testing the neural network
    def prepering_data(self):
        self.training_images = np.stack(self.traning_data['pixels'].values)
        self.training_labels = self.traning_data['emotion'].values
        self.testing_images = np.stack(self.test_data['pixels'].values)
        self.testing_labels = self.test_data['emotion'].values
        self.training_images = self.training_images.reshape(-1, 48, 48, 1)
        self.testing_images = self.testing_images.reshape(-1, 48, 48, 1)

    # Expand the dimensions of a single image to match the input shape expected by the neural network
    def prosses_image(self, image):
    # had to use try as i dident know what the imput image was
        try:
            image = cv2.imread(image)
            image = cv2.resize(image, (48, 48))
                    
                    # Convert to grayscale if it's not already
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Normalize the image
            image = image.astype('float32') / 255.0
                    
                    # Reshape the image to (48, 48, 1)
            image = image.reshape(48, 48, 1)
        finally:
            image = np.expand_dims(image, axis=0)  
            return image
        
# Controles the creating and training off the model
class Modle(FER13Data):
    def __init__(self):
        super().__init__()
        

    # crating my model
    def create_modle(self): 
        super().prepering_data()

        self.model = Sequential()
        self.shape = (48,48,1)
        self.model.add(Conv2D(32,(3,3), activation='relu', input_shape = self.shape, padding = 'same'))
        self.model.add(Conv2D(32,(3,3), activation='relu', padding = "same"))
        self.model.add(Dropout(0.25))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64,(3,3), activation = 'relu', padding = 'same'))
        self.model.add(Conv2D(64,(3,3), activation = 'relu', padding = 'same'))
        self.model.add(Dropout(0.25))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
        self.model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
        self.model.add(Dropout(0.25))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(256, (3,3), activation='relu', padding = 'same'))
        self.model.add(Conv2D(256, (3,3), activation='relu', padding = 'same'))
        self.model.add(Dropout(0.25))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7))
        self.model.add(Activation('softmax'))
        self.model.summary()

    # display the layout of the model
    def get_modle_layout(self):
        self.create_modle()
        self.model.summary()

    
    def train_modle(self):
        self.create_modle()

    #TODO making the modle stop traning if no improvment are found aveter 3 epsolons 
        self.early_stop = EarlyStopping(
            monitor='val_loss', 
            min_delta=0.001, 
            patience=3, 
            verbose=1, 
            restore_best_weights=True)

        # randomising the imiages 
        self.train_datagen = ImageDataGenerator(  
            rotation_range=20,  
            width_shift_range=0.2,  
            height_shift_range=0.2,  
            shear_range=0.2,  
            zoom_range=0.2,  
            horizontal_flip=True,  
            fill_mode='nearest'  
        )

        self.train_generator = self.train_datagen.flow(
            self.training_images,
            self.training_labels,
            batch_size=32,
            shuffle=True
        )

        # traning the modle 
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # training the model and saving it 
        # keeps each version of the accuracy in self.history 
        self.history = self.model.fit(
            self.training_images, 
            self.training_labels, 
            epochs=20, 
            validation_data=(self.testing_images, self.testing_labels),
            callbacks = [self.early_stop],
            shuffle = True)
        
        self.model.save('emotion_detection_modle.h5')

    def load_modle(self):
        self.modle = load_model('emotion_detection_model/emotion_detection_modle.h5')

    # testing the modle and displaying accuracy
    def get_acuracy(self):
        self.load_modle()
        super().prepering_data()
        test_loss, test_accuracy = self.modle.evaluate(self.testing_images, self.testing_labels)
        print(f'the loss is {test_loss} \n the acuracy is {test_accuracy}')

    # takes an imige and uses the model to predict teh output 
    def predict(self,image):
        prosses_image = super().prosses_image(image)
        prediction = self.modle.predict(prosses_image)
        predicted_label = self.emotions[np.argmax(prediction)]
        return(predicted_label)

if __name__ == '__main__':
    modle = Modle()
    modle.get_acuracy()
    