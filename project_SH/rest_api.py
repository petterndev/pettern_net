import os
import requests
import json
import pickle

class APIError(Exception):
    """An API Error Exception"""

    def __init__(self, status):
        self.status = status

    def __str__(self):
        return "APIError: status={}".format(self.status)

token = '6c2d1e42ae4dff1a42253919ca23161fb96d393c'

headers = { 'Authorization' : 'Token ' + token }

a_side = 1
a_top = 1
b_side = 1
b_top = 1
c_side = 1
c_top = 1

if os.path.isfile('objs.pkl'):
    with open('objs.pkl','rb') as f:
        a_side, a_top, b_side, b_top, c_side, c_top = pickle.load(f)


for i in range(69):
    page = i+1
    resp = requests.get("https://www.datamaker.io/api/pettern/pets/?page="+str(page), headers=headers, verify=False)
    if resp.status_code != 200:
        # This means something went wrong.
        raise APIError(resp.status_code)

    for dog in resp.json()['results']:
        if dog['bsc'] < 4:
            for camera_angle in dog['pictures']:
                if camera_angle == []:
                    continue
                else:
                    if camera_angle['view'] == 'side':
                        save_path = './data/side/slim/'
                        if a_side < 10:
                            img_path = os.path.join(save_path,'00000'+str(a_side)+'.jpg')
                        elif a_side < 100:
                            img_path = os.path.join(save_path,'0000'+str(a_side)+'.jpg')
                        else:
                            img_path = os.path.join(save_path,'000'+str(a_side)+'.jpg')
                        f = open(img_path,'wb')
                        f.write(requests.get(camera_angle['file']).content)
                        f.close()
                        a_side += 1
                    else:
                        save_path = './data/top/slim/'
                        if a_top < 10:
                            img_path = os.path.join(save_path,'00000'+str(a_top)+'.jpg')
                        elif a_top < 100:
                            img_path = os.path.join(save_path,'0000'+str(a_top)+'.jpg')
                        else:
                            img_path = os.path.join(save_path,'000'+str(a_top)+'.jpg')
                        f = open(img_path,'wb')
                        f.write(requests.get(camera_angle['file']).content)
                        f.close()
                        a_top += 1
        elif dog['bsc'] < 7:
            for camera_angle in dog['pictures']:
                if camera_angle == []:
                    continue
                else:
                    if camera_angle['view'] == 'side':
                        save_path = './data/side/normal/'
                        if b_side < 10:
                            img_path = os.path.join(save_path,'00000'+str(b_side)+'.jpg')
                        elif b_side < 100:
                            img_path = os.path.join(save_path,'0000'+str(b_side)+'.jpg')
                        else:
                            img_path = os.path.join(save_path,'000'+str(b_side)+'.jpg')
                        f = open(img_path,'wb')
                        f.write(requests.get(camera_angle['file']).content)
                        f.close()
                        b_side += 1
                    else:
                        save_path = './data/top/normal/'
                        if b_top < 10:
                            img_path = os.path.join(save_path,'00000'+str(b_top)+'.jpg')
                        elif b_top < 100:
                            img_path = os.path.join(save_path,'0000'+str(b_top)+'.jpg')
                        else:
                            img_path = os.path.join(save_path,'000'+str(b_top)+'.jpg')
                        f = open(img_path,'wb')
                        f.write(requests.get(camera_angle['file']).content)
                        f.close()
                        b_top += 1
        else:
            for camera_angle in dog['pictures']:
                if camera_angle == []:
                    continue
                else:
                    if camera_angle['view'] == 'side':
                        save_path = './data/side/fat/'
                        if c_side < 10:
                            img_path = os.path.join(save_path,'00000'+str(c_side)+'.jpg')
                        elif c_side < 100:
                            img_path = os.path.join(save_path,'0000'+str(c_side)+'.jpg')
                        else:
                            img_path = os.path.join(save_path,'000'+str(c_side)+'.jpg')
                        f = open(img_path,'wb')
                        f.write(requests.get(camera_angle['file']).content)
                        f.close()
                        c_side += 1
                    else:
                        save_path = './data/top/fat/'
                        if c_top < 10:
                            img_path = os.path.join(save_path,'00000'+str(c_top)+'.jpg')
                        elif c_top < 100:
                            img_path = os.path.join(save_path,'0000'+str(c_top)+'.jpg')
                        else:
                            img_path = os.path.join(save_path,'000'+str(c_top)+'.jpg')
                        f = open(img_path,'wb')
                        f.write(requests.get(camera_angle['file']).content)
                        f.close()
                        c_top += 1

    
with open('objs.pkl', 'wb') as f: 
    pickle.dump([a_side, a_top, b_side, b_top, c_side, c_top], f)