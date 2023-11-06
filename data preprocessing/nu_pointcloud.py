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

def get_lidar_pc(anntoken):
    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])
    lidar = sample_record['data']['LIDAR_TOP']
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(lidar, selected_anntokens=[anntoken])
    LPC = LidarPointCloud.from_file(data_path)
    t_f = points_in_box(boxes[0],LPC.points[:3])
    tmp = LPC.points.T
    subset = []
    for i in range(len(t_f)):
        if t_f[i]==True:
            subset.append(tmp[i])
    subset = np.array(subset)

    return subset.T[:3]
    
    
def save_pts(root,X,y,s_name):
    for i in range(len(y)):
        folder = root+y[i]
        if not os.path.exists(folder):
            os.mkdir(folder)
        with open(folder+'/'+y[i] + '_' + s_name+str(i)+'.txt', 'w') as fp:
            three_cols = []
            while len(three_cols) < 1024:
                for item in X[i].T:
                    three_cols.append(item)
            for j in three_cols:
                # write each item on a new line
                fp.write("{},{},{}\n".format(j[0],j[1],j[2]))
    with open(root+'modelnet40_'+ s_name+'.txt', 'w') as fp:
        for i in range(len(y)):
            tmp = y[i] + '_' + s_name+str(i)
            fp.write("{}\n".format(tmp))
    ls2set = set(y)
    with open(root+'modelnet40_shape_names.txt', 'w') as fp:
        for j in ls2set:
            fp.write("{}\n".format(j))
    return 0

token_list = []

closed_X = []
closed_y = []
open_X = []
open_y = []

openclass = ['vehicle.emergency.ambulance','animal','human.pedestrian.personal_mobility','human.pedestrian.stroller']


with open('') as f:
    token_list = f.read().splitlines()

for i in token_list:
    label = nusc.get('sample_annotation', i)['category_name']
    if (label in openclass):
        open_X.append(get_lidar_pc(i))
        open_y.append(label)
    else:
        closed_X.append(get_lidar_pc(i))
        closed_y.append(label)

X_train, X_test, y_train, y_test = train_test_split(closed_X, closed_y,stratify=closed_y, test_size=0.2, random_state=42)
    
print('y_train',Counter(y_train))
print('y_test',Counter(y_test))
print('open_y',Counter(open_y))


save_pts(r'',open_X,open_y,'test')
save_pts(r'',X_test,y_test,'test')
save_pts(r'',X_train,y_train,'train')