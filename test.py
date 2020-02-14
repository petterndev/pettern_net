# import matplotlib.pyplot as plt
# import pathlib
# from tensorflow.keras.models import load_model
# import cv2
# model = load_model('test_model_1.h5')

# train_data_dir = 'modified_crop/side/train/'
# train_data_dir = pathlib.Path(train_data_dir)
# a=0
# b=0
# c=0
# for train in list(train_data_dir.glob('normal/*.jpg')):
#     image = plt.imread(str(train))
#     if (image.shape[0]==100):
#         img = image.astype('float32')/255
#         img2 = img.reshape(1,224,100,3)
#         if model.predict_classes(img2)==0:
#             a+=1
#         elif model.predict_classes(img2)==1:
#             b+=1
#         else:
#             c+=1
# print('slim: ', str(a))
# print('normal: ', str(b))
# print('fat: ', str(c))


import matplotlib.pyplot as plt
import pathlib
from tensorflow.keras.models import load_model
import cv2
import os
model = load_model('test_model_1.h5')


image = plt.imread('dog.jpg')
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
print(img.shape)
path = "./"
img_name = os.path.join(path,'resize.jpg')
cv2.imwrite(img_name, img)
print("done")

if (img.shape[0]==100):
    img = img.astype('float32')/255
    img2 = img.reshape(1,224,100,3)
    print(model.predict_classes(img2))
else:
    print(image.shape[0])
    print("error")