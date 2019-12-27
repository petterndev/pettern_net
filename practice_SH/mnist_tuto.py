from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# print('x_train shape:', train_images.shape)
# print(train_images.shape[0], 'train samples')
# print(test_images.shape[0], 'test samples')

tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs")

model = keras.Sequential([
    keras.layers.Reshape((28,28,1), input_shape=train_images.shape[1:]),
    keras.layers.Conv2D(32, (3,3), strides=(1,1), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None),
    keras.layers.Dropout(0.3),
    keras.layers.Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None),
    keras.layers.Dropout(0.3),
    keras.layers.Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None),
    keras.layers.Dropout(0.3),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10,callbacks=[tensorboard_callback])


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)