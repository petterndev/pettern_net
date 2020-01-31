#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2
import argparse
import numpy as np
import os
import tensorflow as tf
import pathlib


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


train_data_dir = '/Users/kevin/Desktop/KAIST/CUOP/project/data_added/side/val/'
train_data_dir = pathlib.Path(train_data_dir)
list_ds = list(train_data_dir.glob('slim/*.jpg'))

for img_path in list_ds:
    img_path = str(img_path)
    print(img_path)
    image = cv2.imread(img_path)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open("yolov3.txt", 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        if class_ids[i] == 16:
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            #draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
            print(x,y)
            print(w,h)

            if x<0:
                x=0
            if y<0:
                y=0
            img = image[round(y):round(y + h) , round(x):round(x + w)]
            base_width = 504
            print(img.shape)
            ratio = base_width/float(img.shape[1])
            new_height = int(float(img.shape[0])*float(ratio))
            img = cv2.resize(img,(base_width,new_height))
            # img = tf.image.resize(img,[224,224], preserve_aspect_ratio= True)
            path = "/Users/kevin/Desktop/KAIST/CUOP/project/data_yolo/side/val/slim/"
            jpg_name = img_path.split("/")[-1]
            img_name = os.path.join(path,jpg_name)
            cv2.imwrite(img_name, np.float32(img))
        