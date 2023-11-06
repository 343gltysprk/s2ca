import argparse
import numpy as np
import utils
from calibration import Calibration
import os

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


points_path0 = 'kitti/training/velodyne/{}.bin'.format(args.idx)
label_path0 = 'kitti/training/label_2/{}.txt'.format(args.idx)
calib_path0 = 'kitti/training/calib/{}.txt'.format(args.idx)
path = 'output'

def kitti_parse(save_path,points_path,label_path,calib_path, tr, te):
    calib = Calibration(calib_path)
    points = utils.load_point_clouds(points_path)
    bboxes, names = utils.load_3d_boxes(label_path)
    bboxes = calib.bbox_rect_to_lidar(bboxes)

    corners3d = utils.boxes_to_corners_3d(bboxes)
    points_flag = utils.is_within_3d_box(points, corners3d)
    
    known = ['Car', 'Pedestrian', 'Van', 'Cyclist']
    unknown = ['Truck', 'Misc', 'Tram', 'Person_sitting']

    points_is_within_3d_box = []
    for i in range(len(points_flag)):
        p = points[points_flag[i]]
        if len(p)>0:
            points_is_within_3d_box.append(p)
            box = bboxes[i]
            points_canonical, box_canonical = utils.points_to_canonical(p, box)
            points_canonical, box_canonical = utils.lidar_to_shapenet(points_canonical, box_canonical)
            if names[i] in known:
                out = save_path + '/closed'
            else:
                out = save_path + '/open'
            if not os.path.exists(out):
                os.mkdir(out)
            if not os.path.exists(out+'/'+names[i]):
                os.mkdir(out+'/'+names[i])
            pts_name = '{}/{}/{}_{}point{}.txt'.format(out,names[i],names[i], args.idx, i)
            if names[i] in known:
                tr.append('{}_{}point{}'.format(names[i], args.idx, i))
            else:
                te.append('{}_{}point{}'.format(names[i], args.idx, i))
            utils.write_points(points_canonical, pts_name)
            
with open('k.txt') as f:
    k_list = f.read().splitlines()

with open('u.txt') as f:
    u_list = f.read().splitlines()

log_tr = []
log_te = []
log_ood = []
empty = []


for i in u_list:
    args.idx = i
    points_path0 = 'kitti/training/velodyne/{}.bin'.format(i)
    label_path0 = 'kitti/training/label_2/{}.txt'.format(i)
    calib_path0 = 'kitti/training/calib/{}.txt'.format(i)
    path = 'output'
    kitti_parse(path,points_path0,label_path0,calib_path0,log_te,log_ood)

for i in k_list:
    args.idx = i
    points_path0 = 'kitti/training/velodyne/{}.bin'.format(i)
    label_path0 = 'kitti/training/label_2/{}.txt'.format(i)
    calib_path0 = 'kitti/training/calib/{}.txt'.format(i)
    path = 'output'
    kitti_parse(path,points_path0,label_path0,calib_path0,log_tr,empty)

print(empty)


with open('modelnet40_train.txt', 'w') as fp:
    for j in log_tr:
        fp.write("{}\n".format(j))
        
with open('modelnet40_test.txt', 'w') as fp:
    for j in log_te:
        fp.write("{}\n".format(j))
        
with open('modelnet40_ood.txt', 'w') as fp:
    for j in log_ood:
        fp.write("{}\n".format(j))
