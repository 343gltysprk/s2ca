from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from matplotlib import rcParams
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix, points_in_box
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
    if not os.path.exists(save_path+'training-set'):
        os.mkdir(save_path+'training-set')
    if not os.path.exists(save_path+'test-set'):
        os.mkdir(save_path+'test-set')
    if not os.path.exists(save_path+'test-open'):
        os.mkdir(save_path+'test-open')
    openclass = ['vehicle.emergency.ambulance','animal','human.pedestrian.personal_mobility','human.pedestrian.stroller']
    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])
    lidar = sample_record['data']['LIDAR_TOP']
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(lidar, selected_anntokens=[anntoken])
    LPC = LidarPointCloud.from_file(data_path)
    t_f = points_in_box(boxes[0],LPC.points[:3])
    H=128
    W=2048
    proj_fov_up=10.0
    proj_fov_down=-30.0
    # laser parameters
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(lidar, selected_anntokens=[anntoken])
    t_f = points_in_box(boxes[0],LPC.points[:3])
    point_cloud = LPC.points.T
    fov_up = proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad
    
    # get depth of all points
    depth = np.linalg.norm(point_cloud, 2, axis=1)
    # get scan components
    scan_x = point_cloud[:, 1]
    scan_y = point_cloud[:, 0]
    scan_z = point_cloud[:, 2]


    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov

    proj_x *= W                              # in [0.0, W]
    proj_y *= H                              # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]
    t_f = t_f[order]
    
    obj_x =  proj_x[t_f]
    obj_y =  proj_y[t_f]
    proj_range = np.full((H, W), 0,dtype=np.float32)
    
    proj_range[proj_y, proj_x] = depth
    a = min(obj_y)
    b = max(obj_y)
    c = min(obj_x)
    d = max(obj_x)
    cropped_image = proj_range[a:b,c:d]
    #cropped_image  = cv2.cvtColor(obj,cv2.COLOR_GRAY2RGB)
    
    h,w = cropped_image.shape
    if h==0 or w==0:
        return False

    cropped_image = Image.fromarray(cropped_image)
    cropped_image = expand2square(cropped_image,0).resize((32, 32))
    cropped_image = cropped_image.convert('RGB')
    #print(cropped_image)
    ann_record = nusc.get('sample_annotation', anntoken)
    class_name = ann_record['category_name']
    #x = random.random() 
    if (class_name in openclass):
        if class_name in openclass:
            #class_name = 'unknown'
            image_path = save_path + 'test-open/'+ class_name
            if not os.path.exists(image_path):
                os.mkdir(image_path)
            filename = image_path + '/' + anntoken + '.png'
            cropped_image.save(filename)
            return True
            
    elif is_test:
        image_path = save_path + 'test-set/'+ class_name
        if not os.path.exists(image_path):
            os.mkdir(image_path)
        filename = image_path + '/' + anntoken + '.png'
        cropped_image.save(filename)
        return True
    
    image_path = save_path + 'training-set/'+ class_name
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    filename = image_path + '/' + anntoken + '.png'
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