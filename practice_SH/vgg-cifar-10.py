from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

cifar = keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifar.load_data()

#Pre-Processing

img_data = keras.preprocessing.image.ImageDataGenerator(
    rotation_range = 90,
    horizontal_flip = True,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    featurewise_center = True,
    featurewise_std_normalization = True
)

img_data.fit(train_images)

tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs")

vgg_model = keras.applications.VGG16(input_shape=train_images.shape[1:], include_top=False, weights='imagenet')
vgg_model.trainable = False


model = keras.Sequential([
    vgg_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(
    img_data.flow(train_images,train_labels),
    epochs = 10,
    callbacks=[tensorboard_callback]
)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)