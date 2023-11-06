import argparse
import numpy as np
import utils
from calibration import Calibration
import os
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

parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--idx', type=str, default='000003',
                    help='specify data index: {idx}.bin')
parser.add_argument('--category', type=str, default='car',
                    help='specify the category to be extracted,' + 
                        '{ \
                            Car, \
                            Van, \
                            Truck, \
                            Pedestrian, \
                            Person_sitting, \
                            Cyclist, \
                            Tram \
                        }')
args = parser.parse_args()

with open('k.txt') as f:
    k_list = f.read().splitlines()

with open('u.txt') as f:
    u_list = f.read().splitlines()


def expand2square(pil_img, background_color=(0, 0, 0)):
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
    

def parse_img(save_path,token,val=False):
    global_name = 0
    unknown = ['Truck', 'Misc', 'Tram', 'Person_sitting']
    if not os.path.exists(save_path+'training-set'):
        os.mkdir(save_path+'training-set')
    if not os.path.exists(save_path+'test-set'):
        os.mkdir(save_path+'test-set')
    if not os.path.exists(save_path+'test-open'):
        os.mkdir(save_path+'test-open')
        
    img_path = 'kitti/training/image_2/{}.png'.format(i)
    label_path0 = 'kitti/training/label_2/{}.txt'.format(i)
    
    with open(label_path0, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        arr = line.strip().split(' ')
        im = cv2.imread(img_path,cv2.IMREAD_COLOR)
        cropped_image = im[int(float(arr[5])):int(float(arr[7])),int(float(arr[4])):int(float(arr[6]))]
        h,w,chan = cropped_image.shape
        if h==0 or w==0:
            print(token)
            continue
        
        saved_d = None
        if arr[0] in unknown:
            saved_d = save_path + 'test-open/' + arr[0]
        elif val==True:
            saved_d = save_path + 'test-set/' + arr[0]
        else:
            saved_d = save_path + 'training-set/' + arr[0]
            
        if not os.path.exists(saved_d):
            os.mkdir(saved_d)

        filename = saved_d + '/' + token + '_' + str(global_name) + '.png'
        global_name+=1
        cropped_image = Image.fromarray(cropped_image, 'RGB')
        cropped_image = expand2square(cropped_image,(0, 0, 0)).resize((32, 32))
        cropped_image.save(filename)


for i in u_list:
    parse_img('kitti32/',i,val=True)

for i in k_list:
    parse_img('kitti32/',i,val=False)