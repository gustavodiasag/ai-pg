import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

training_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)

training_set = training_datagen.flow_from_directory(
    'training_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary')

validation_set = validation_datagen.flow_from_directory(
    'test_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary')

steps_per_epoch = training_set.samples // training_set.batch_size
validation_steps = validation_set.samples // validation_set.batch_size

history = classifier.fit(
    training_set,
    steps_per_epoch = steps_per_epoch,
    epochs = 100,
    validation_data = validation_set,
    validation_steps = validation_steps)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist['val_loss']
hist.describe()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()

plt.clf()
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

def plot_prediction(filename):
    test_image = image.load_img(filename, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    
    result = classifier.predict(test_image)
    training_set.class_indices

    if result[0][0] == 1:
        prediction = 'Homer'
    else:
        prediction = 'Bart'

    plt.imshow(mpimg.imread(filename))
    plt.title(prediction)
    plt.show()

plot_prediction('test_set/bart/bart143.bmp')
plot_prediction('test_set/homer/homer9.bmp')
plot_prediction('test_set/bart/bart6.bmp')
plot_prediction('test_set/homer/homer20.bmp')