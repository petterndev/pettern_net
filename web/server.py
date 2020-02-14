# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from tensorflow.keras.models import load_model
import cv2
import os


app = Flask(__name__)

model = load_model('../test_model_1.h5')

# # 플레이스 홀더를 설정합니다.
# X = tf.placeholder(tf.float32, shape=[None, 4])
# Y = tf.placeholder(tf.float32, shape=[None, 1])

# W = tf.Variable(tf.random_normal([4, 1]), name="weight")
# b = tf.Variable(tf.random_normal([1]), name="bias")

# # 가설을 설정합니다.
# hypothesis = tf.matmul(X, W) + b

# # 저장된 모델을 불러오는 객체를 선언합니다.
# saver = tf.train.Saver()
# model = tf.global_variables_initializer()

# # 세션 객체를 생성합니다.
# sess = tf.Session()
# sess.run(model)

# # 저장된 모델을 세션에 적용합니다.
# save_path = "./model/saved.cpkt"
# saver.restore(sess, save_path)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        # 파라미터를 전달 받습니다.
        dog = request.files['dog']
        dog.save(f'static/uploads/{secure_filename(dog.filename)}')

        # filename = 'uploads/{secure_filename(dog.filename)}'
        # print("--------------------------------", filename)
        dog_name = dog.filename
        dog_path = "./static/uploads"
        dog_path2 = "./uploads"
        dog2 = os.path.join(dog_path,dog_name)
        dog3 = os.path.join(dog_path2,dog_name)
        print("------------------------------", dog3)

        image = plt.imread(dog2)
        Width = image.shape[1]
        Height = image.shape[0]
        q = 224/Width
        img = cv2.resize(image, (224, round(Height*q)))
        Height_img = img.shape[0]
        if round(Height_img/2)-50 < 0:
            img = img[:100 , :]
            print("이미지가 너무 작아요ㅜ")
        else:
            img = img[round(Height_img/2)-50:round(Height_img/2)+50 , :]
        # print(img.shape)

        # avg_temp = float(request.form['avg_temp'])
        # min_temp = float(request.form['min_temp'])
        # max_temp = float(request.form['max_temp'])
        # rain_fall = float(request.form['rain_fall'])

        # 배추 가격 변수를 선언합니다.
        bcs = 5

        # 입력된 파라미터를 배열 형태로 준비합니다.
        # data = ((avg_temp, min_temp, max_temp, rain_fall), (0, 0, 0, 0))
        # arr = np.array(data, dtype=np.float32)

        # 입력 값을 토대로 예측 값을 찾아냅니다.
        # x_data = arr[0:4]
        # dict = sess.run(hypothesis, feed_dict={X: x_data})
            
        # 결과 배추 가격을 저장합니다.
        if (img.shape[0]==100):
            img = img.astype('float32')/255
            img2 = img.reshape(1,100,224,3)
            bcs = model.predict_classes(img2)
            if bcs==0:
                bcs = "fat"
            elif bcs==1:
                bcs = "normal"
            else:
                bcs = "slim"
        else:
            print(image.shape[0])
            print("error")
        print(bcs)
        
        return render_template('index.html', bcs=bcs, image = dog3)

host_addr = "143.248.212.80"
port_num = "30583"

if __name__ == '__main__':
   app.run(debug = True)