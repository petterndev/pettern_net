import requests
from PIL import Image
from urllib.request import urlopen
from io import BytesIO
import cv2
import numpy
import json
header = {'Authorization': 'Token 6c2d1e42ae4dff1a42253919ca23161fb96d393c'}

#print(r.json())

#print(json_result)
#print(len(json_result))
a=0
b=0
for j in range(70):
    r = requests.get('https://www.datamaker.io/api/pettern/pets/?page=' + str(j+1), headers=header)
    json_data = r.json()
    json_result = json_data["results"]
    for i in range(len(json_result)):
        json_picture = json_result[i]['pictures']
        json_bcs = json_result[i]['bsc']
        if json_picture != []:
            if len(json_picture)==2:
                top = json_picture[1]['file']
                response_top = requests.get(top)
                img_top = Image.open(BytesIO(response_top.content))
                opencvImage_top = cv2.cvtColor(numpy.array(img_top), cv2.COLOR_RGB2BGR)
                if 1<=json_bcs<=3:
                    path = 'top_1/'
                elif 4<=json_bcs<=6:
                    path = 'top_2/'
                else:
                    path = 'top_3/'
                print(img_top)
                cv2.imwrite(str(path) + str(b) + '.jpg',opencvImage_top)
                b+=1
            side = json_picture[0]['file']
            response_side = requests.get(side)
            img_side = Image.open(BytesIO(response_side.content))
            opencvImage_side = cv2.cvtColor(numpy.array(img_side), cv2.COLOR_RGB2BGR)
            if 1<=json_bcs<=3:
                path = 'side_1/'
            elif 4<=json_bcs<=6:
                path = 'side_2/'
            else:
                path = 'side_3/'
            print(img_side)
            cv2.imwrite(str(path) + str(a) + '.jpg',opencvImage_side)
            a+=1
                