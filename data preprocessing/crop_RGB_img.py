from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from matplotlib import rcParams
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
import numpy as np
from PIL import Image
import cv2
import os
import random
from sklearn.model_selection import train_test_split
from collections import Counter

nusc = NuScenes(version='v1.0-trainval', dataroot=r'', verbose=True)

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def crop_image_from_annotation(anntoken,save_path,is_test=False):
    openclass = ['vehicle.emergency.ambulance','animal','human.pedestrian.personal_mobility','human.pedestrian.stroller']
    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])
    if not os.path.exists(save_path+'training-set'):
        os.mkdir(save_path+'training-set')
    if not os.path.exists(save_path+'test-set'):
        os.mkdir(save_path+'test-set')
    if not os.path.exists(save_path+'test-open'):
        os.mkdir(save_path+'test-open')
    assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'
    boxes, cam = [], []
    cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
    
    for cam in cams:
        str1, boxes, arr1 = nusc.get_sample_data(sample_record['data'][cam],selected_anntokens=[anntoken])
        if len(boxes)>0:
            break
                
    if (len(boxes)==0):
        print(anntoken)
        return False
    
    cam = sample_record['data'][cam]
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam, selected_anntokens=[anntoken])
    im = cv2.imread(data_path,cv2.IMREAD_COLOR)
    corners = view_points(boxes[0].corners(), camera_intrinsic, normalize=True)[:2, :]
    h,w,chan = im.shape
    a = int(min(corners[1]))
    a = max(0,a)
    b = int(max(corners[1]))
    b = min(h,b)
    c = int(min(corners[0]))
    c = max(0,c)
    d = int(max(corners[0]))
    d = min(w,d)
    cropped_image = im[a:b,c:d]
    h,w,chan = cropped_image.shape
    if h==0 or w==0:
        return False
    cropped_image = im[a:b,c:d]
    cropped_image = Image.fromarray(cropped_image, 'RGB')
    cropped_image = expand2square(cropped_image,(0, 0, 0)).resize((32, 32))
    ann_record = nusc.get('sample_annotation', anntoken)
    class_name = ann_record['category_name']
    #x = random.random() 
    if (class_name in openclass):
        if class_name in openclass:
            #class_name = 'unknown'
            image_path = save_path + 'test-open/'+ class_name
            if not os.path.exists(image_path):
                os.mkdir(image_path)
            filename = image_path + '/' + anntoken + '.jpg'
            cropped_image.save(filename)
            return True
            
    elif is_test:
        image_path = save_path + 'test-set/'+ class_name
        if not os.path.exists(image_path):
            os.mkdir(image_path)
        filename = image_path + '/' + anntoken + '.jpg'
        cropped_image.save(filename)
        return True
    
    image_path = save_path + 'training-set/'+ class_name
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    filename = image_path + '/' + anntoken + '.jpg'
    #cv2.imwrite(filename, cropped_image)
    cropped_image.save(filename)
    return True
    


token_list = []

closed_X = []
closed_y = []
open_X = []
open_y = []

openclass = ['vehicle.emergency.ambulance','animal','human.pedestrian.personal_mobility','human.pedestrian.stroller']
not_visible = []

with open('') as f:
    token_list = f.read().splitlines()

for i in token_list:
    label = nusc.get('sample_annotation', i)['category_name']
    if (label in openclass):
        open_X.append(i)
        open_y.append(label)
    else:
        closed_X.append(i)
        closed_y.append(label)

X_train, X_test, y_train, y_test = train_test_split(closed_X, closed_y,stratify=closed_y, test_size=0.2, random_state=42)
    
print('y_train',Counter(y_train))
print('y_test',Counter(y_test))
print('open_y',Counter(open_y))

for i in open_X:
    if not crop_image_from_annotation(i,''):
        not_visible.append(i)

for i in X_train:
    if not crop_image_from_annotation(i,''):
        not_visible.append(i)

for i in X_test:
    if not crop_image_from_annotation(i,'',is_test=True):
        not_visible.append(i)