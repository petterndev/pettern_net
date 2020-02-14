from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import cv2

AUTOTUNE = tf.data.experimental.AUTOTUNE

keras = tf.keras

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2]

#------------------------------------------crop---------------------------------
# train_data_dir = './modified_resize/side/train/'
# train_data_dir = pathlib.Path(train_data_dir)
# test_data_dir = './modified_resize/side/val/'
# test_data_dir = pathlib.Path(test_data_dir)
# a=0
# b=0
# c=0
# for train in list(test_data_dir.glob('*/*.jpg')):
#   print(train)
#   image = cv2.imread(str(train))
#   print(image.shape)
#   Width = image.shape[1]
#   Height = image.shape[0]
#   img = image[round(Height/2)-50:round(Height/2)+50 , :]
#   print(img.shape)
#   if get_label(str(train)) == "slim":
#     path = "./modified_crop/side/val/slim"
#     img_name = os.path.join(path,str(a)+".jpg")
#     a+=1
#   elif get_label(str(train)) == "normal":
#     path = "./modified_crop/side/val/normal"
#     img_name = os.path.join(path,str(b)+".jpg")
#     b+=1
#   else:
#     path = "./modified_crop/side/val/fat"
#     img_name = os.path.join(path,str(c)+".jpg")
#     c+=1
#   cv2.imwrite(img_name, img)

#------------------------------------------resize---------------------------------
train_data_dir = './modified_resize/side/train/'
train_data_dir = pathlib.Path(train_data_dir)
test_data_dir = './modified_resize/side/val/'
test_data_dir = pathlib.Path(test_data_dir)
a=0
b=0
c=0
n=0
# shortest = 224
for train in list(test_data_dir.glob('slim/*.jpg')):
  if n<10:
    print(train)
    image = cv2.imread(str(train))
    print(image.shape)
    Width = image.shape[1]
    Height = image.shape[0]
    q = 224/Width
    img = cv2.resize(image, (224, round(Height*q)))
    Height_img = img.shape[0]
    if round(Height_img/2)-25 < 0:
      img = img[:100 , :]
      print("이미지가 너무 작아요ㅜ")
    else:
      img = img[round(Height_img/2)-25:round(Height_img/2)+75 , :]
    print(img.shape)
    #img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_not = cv2.bitwise_not(img)
    # if img.shape[0]<shortest:
    #   shortest = img.shape[0]
    #   print("UPDATE SHORTEST", shortest)
    if get_label(str(train)) == "slim":
      print("kdjfnsdjnj")
      path = "./modified_crop/side/val/slim"
      img_name = os.path.join(path,"not"+str(a)+".jpg")
      a+=1
    elif get_label(str(train)) == "normal":
      path = "./modified_crop/side/val/normal"
      img_name = os.path.join(path,"not"+str(b)+".jpg")
      b+=1
    else:
      path = "./modified_crop/side/val/fat"
      img_name = os.path.join(path,"not"+str(c)+".jpg")
      c+=1
    cv2.imwrite(img_name, img_not)
    print(n)
    n+=1
# print("------------------------------------------SHORTEST:", shortest)

#-----------------------------------------object detection---------------------------------
# train_data_dir = './data_modified_copy/side/train/'
# train_data_dir = pathlib.Path(train_data_dir)
# test_data_dir = './data_modified_copy/side/val/'
# test_data_dir = pathlib.Path(test_data_dir)
# a = 0
# b = 0
# c = 0
# for train in list(test_data_dir.glob('*/*.jpg')):
#   print(train)
#   image = cv2.imread(str(train))
#   print(image.shape)

#   Width = image.shape[1]
#   Height = image.shape[0]
#   scale = 0.00392

#   classes = None

#   with open("yolov3.txt", 'r') as f:
#       classes = [line.strip() for line in f.readlines()]

#   COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

#   net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

#   blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

#   net.setInput(blob)

#   outs = net.forward(get_output_layers(net))

#   class_ids = []
#   confidences = []
#   boxes = []
#   conf_threshold = 0.5
#   nms_threshold = 0.4
#   for out in outs:
#     for detection in out:
#         scores = detection[5:]
#         class_id = np.argmax(scores)
#         confidence = scores[class_id]
#         if confidence > 0.5:
#             center_x = int(detection[0] * Width)
#             center_y = int(detection[1] * Height)
#             w = int(detection[2] * Width)
#             h = int(detection[3] * Height)
#             x = center_x - w / 2
#             y = center_y - h / 2
#             class_ids.append(class_id)
#             confidences.append(float(confidence))
#             boxes.append([x, y, w, h])


#   indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
#   #print(indices)
#   for i in indices:
#       i = i[0]
#       if class_ids[i]==16:
#         box = boxes[i]
#         x = box[0]
#         y = box[1]
#         w = box[2]
#         h = box[3]
#         print(round(x), round(y), round(h), round(w))
#         if x<0:
#           x=0
#         if y<0:
#           y=0
#         img = image[round(y):round(y + h) , round(x):round(x + w)]
#         #img = cv2.resize(img, dsize=(32, 32))
#         if get_label(str(train)) == "slim":
#           path = "./modified/side/val/slim"
#           img_name = os.path.join(path,str(a)+".jpg")
#           a+=1
#         elif get_label(str(train)) == "normal":
#           path = "./modified/side/val/normal"
#           img_name = os.path.join(path,str(b)+".jpg")
#           b+=1
#         else:
#           path = "./modified/side/val/fat"
#           img_name = os.path.join(path,str(c)+".jpg")
#           c+=1
#         cv2.imwrite(img_name, img)
#       else:
#         print("노퍼피ㅠㅠ")


