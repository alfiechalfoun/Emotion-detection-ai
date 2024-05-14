# 60 pecent acuracy
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D,BatchNormalization 
from tensorflow.keras.metrics import categorical_accuracy 
from tensorflow.keras.callbacks import ModelCheckpoint 
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.optimizers import * 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
 
tf.config.threading.set_intra_op_parallelism_threads(5)
tf.config.threading.set_inter_op_parallelism_threads(3)

if tf.config.experimental.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

def loading_data():
    # lableing
    emotions = ['angry','disgust','fear','happy','neutral','sad','surprise']
    FER_path = '/Users/alfie/Documents/school /computer science /courswork /FER-2013 data set /fer2013.csv'

    #loading the FER_2013 daterset 
    faces = pd.read_csv(FER_path)
    faces['pixels'] = faces['pixels'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape(48, 48) / 255.0)

    # splitting the data into traning and testing 
    traning_data = faces[faces['Usage'] == 'Training'] 
    test_data =  faces[faces['Usage'] == 'PublicTest']

    # prpering the data
    training_images = np.stack(traning_data['pixels'].values)
    training_labels = traning_data['emotion'].values
    testing_images = np.stack(test_data['pixels'].values)
    testing_labels = test_data['emotion'].values
    training_images = training_images.reshape(-1, 48, 48, 1)
    testing_images = testing_images.reshape(-1, 48, 48, 1)
    
    return(training_images, training_labels, testing_images, testing_labels)


# crating my model
def create_modle(): 
    model = Sequential()
    shape = (48,48,1)
    model.add(Conv2D(32, (3,3), activation='relu', input_shape = shape, padding = 'same'))
    model.add(Conv2D(32,(3,3), activation='relu', padding = "same"))
    model.add(Conv2D(32,(3,3),activation='relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3), activation='relu', padding = 'same'))
    model.add(Conv2D(64,(3,3), activation = 'relu',padding = 'same'))
    model.add(Conv2D(64,(3,3), activation ='relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), activation ='relu'))
    model.add(Conv2D(128, (3,3), activation = 'relu', padding = 'same'))
    model.add(Conv2D(128,(3,3), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7))
    model.add(Activation('softmax'))
    return(model)

def train_modle(model):
    training_images, training_labels, testing_images, testing_labels = loading_data()
# making the modle stop traning if no improvment are found aveter 3 epsolons 
    early_stop = EarlyStopping(
        monitor='val_loss', 
        min_delta=0.001, 
        patience=3, 
        verbose=1, 
        restore_best_weights=True)

    # randomising the imiages 
    train_datagen = ImageDataGenerator(  
        rotation_range=20,  
        width_shift_range=0.2,  
        height_shift_range=0.2,  
        shear_range=0.2,  
        zoom_range=0.2,  
        horizontal_flip=True,  
        fill_mode='nearest'  
    )

    train_generator = train_datagen.flow(
        training_images,
        training_labels,
        batch_size=32,
        shuffle=True
    )
    # traning the modle 
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        training_images, 
        training_labels, 
        epochs=20, 
        validation_data=(testing_images, testing_labels),
        callbacks = [early_stop],
        shuffle = True)

    # testing the modle 
    test_loss, test_accuracy = model.evaluate(testing_images, testing_labels)
    print(f'the loss is {test_loss} \n the acuracy is {test_accuracy}')

# saving modle 


if __name__ == '__main__':
    modle = create_modle()
    train_modle(modle)
    model.save('emotion_detection_modle.h5')
