from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
AUTOTUNE = tf.data.experimental.AUTOTUNE

keras = tf.keras



train_data_dir = './data_yolo/side/train/'
# train_data_dir = './hymenoptera_data/train/'
train_data_dir = pathlib.Path(train_data_dir)
test_data_dir = './data_yolo/side/val/'
# test_data_dir = './hymenoptera_data/val/'
test_data_dir = pathlib.Path(test_data_dir)


image_count = len(list(train_data_dir.glob('*/*.jpg')))
image_count_test = len(list(test_data_dir.glob('*/*.jpg')))
print(image_count)
print(image_count_test)

CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*') if item.name != ".DS_Store"])
print(CLASS_NAMES)

HOR_IMG_SIZE = 200
VER_IMG_SIZE = 504

list_ds = tf.data.Dataset.list_files(str(train_data_dir/'*/*.jpg'))
test_list_ds = tf.data.Dataset.list_files(str(test_data_dir/'*/*.jpg'))

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.image.per_image_standardization(img)
  #img = tf.image.resize(img, [HOR_IMG_SIZE, VER_IMG_SIZE])
  # resize the image to the desired size.
  return img

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_labeled_ds = test_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
train_ds = np.empty((image_count,HOR_IMG_SIZE,VER_IMG_SIZE,3))
label_train = np.empty((image_count,3,))
test_ds = np.empty((image_count_test,HOR_IMG_SIZE,VER_IMG_SIZE,3))
label_test = np.empty((image_count_test,3,))

i = 0
j = 0
for train_image, train_label in iter(labeled_ds):
    y_init = int(train_image.shape[0]*0.5 - 100)
    train_image = tf.image.crop_to_bounding_box(
      train_image,
      offset_height=y_init,
      offset_width=0,
      target_height=200,
      target_width=VER_IMG_SIZE
    )
    train_ds[i] = train_image
    label_train[i] = train_label
    i += 1
    # train_ds.append(train_image)
    # train_label_ds.append(train_label)
for test_image, test_label in iter(test_labeled_ds):
    y_init = int(test_image.shape[0]*0.5 - 100)
    test_image = tf.image.crop_to_bounding_box(
      test_image,
      offset_height=y_init,
      offset_width=0,
      target_height=200,
      target_width=VER_IMG_SIZE
    )
    test_ds[j] = test_image
    label_test[j] = test_label
    j += 1

train_ds = tf.keras.utils.normalize(train_ds, axis=1)
test_ds = tf.keras.utils.normalize(test_ds, axis=1)


IMG_SHAPE = (HOR_IMG_SIZE, VER_IMG_SIZE, 3)

tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs")

# Create the base model from the pre-trained model ResNet50
base_model = tf.keras.applications.InceptionResNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

for layer in base_model.layers:
  layer.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
dropout_layer = tf.keras.layers.Dropout(0.5)
prediction_layer = keras.layers.Dense(len(CLASS_NAMES), activation='softmax')

tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs")

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  dropout_layer,
  prediction_layer
])


initial_epochs = 10
validation_steps = 1

print('test')

model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, label_train, epochs=10,callbacks=[tensorboard_callback])

loss0,accuracy0 = model.evaluate(test_ds, label_test, steps = validation_steps, verbose=2)

print("Test loss: {:.2f}".format(loss0))
print("Test accuracy: {:.2f}".format(accuracy0))

model.save('test_model_1.h5') 