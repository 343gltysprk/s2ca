# S2CA: Supervised Contrastive Class Anchor Learning for Open Set Object Recognition in Driving Scenes

This repository is an implementation of S2CA framework, including code for data preprocessing, training and evaluation.
<p align="center">
  <img src="figures/s2ca.png" width="700">
</p>

## CIFAR100 Evaluation
Code and Checkpoint Coming Soon

The codebase is modified from https://github.com/deeplearning-wisc/cider/tree/master

To evaluate S2CA on OOD detection benchmarks

```
cd kitti_pt.py cifar100benchmarks
```

### Data Preparation


OOD datasets can be downloaded via the following links (source: [ATOM](https://github.com/jfc43/informative-outlier-mining/blob/master/README.md)):

* [SVHN](http://ufldl.stanford.edu/housenumbers/test_32x32.mat): download it and place it in the folder of `datasets/small_OOD_dataset/svhn`. Then run `python utils/select_svhn_data.py` to generate test subset.
* [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz): download it and place it in the folder of `datasets/small_OOD_dataset/dtd`.
* [Places365](http://data.csail.mit.edu/places/places365/test_256.tar): download it and place it in the folder of `datasets/ood_datasets/places365/test_subset`. We randomly sample 10,000 images from the original test dataset. 
* [LSUN](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz): download it and place it in the folder of `datasets/small_OOD_dataset/LSUN`.
* [LSUN-resize](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz): download it and place it in the folder of `datasets/small_OOD_dataset/LSUN_resize`.
* [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz): download it and place it in the folder of `datasets/small_OOD_dataset/iSUN`.

For example, run the following commands in the **root** directory to download **LSUN-C**:
```
cd datasets/small_OOD_dataset
wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
tar -xvzf LSUN.tar.gz
```

The directory structure looks like this:

```python
datasets/
---CIFAR10/
---CIFAR100/
---small_OOD_dataset/
------dtd/
------iSUN/
------LSUN/
------LSUN_resize/
------places365/
------SVHN/
```


### Model training

```
sh scripts/train_s2ca_cifar100.sh
```


## Environments

The code was developed with Python 3.9.13 and Pytorch 2.0.1. CUDA is required.

Required packages are specified in [requirements.txt](requirements.txt)

Simply use the following command to install all the packages listed.
```
pip install -r requirements.txt
```

## Datasets
The project involves two large-scale autonomous driving datasets: KITTI and nuScenes. These two datasets are public and can be downloaded from their websites.

[KITTI](https://www.cvlibs.net/datasets/kitti/index.php)

[nuScenes](https://www.nuscenes.org/)

## Quick Start

This project contains a variety of encoder networks and datasets. We now take DGCNN as an example to show how to use S2CA for Open Set Object Recognition.

First, download the KITTI dataset and decompress it to the `data preprocessing` folder.

Organize the downloaded files as follows:
```
├── kitti
│   │── ImageSets
│   │── training
│   │   ├──calib & velodyne & label_2 & image_2
│   │── testing
│   │   ├──calib & velodyne & image_2
│   
├── extract_partial_point_clouds_from_kitti.py
├── utils.py
├── visualize.py
├── calibration.py
```

To extract point cloud for each instances, run:
```
python kitti_pt.py
```

Then we can start Model Training.

```
python training.py --dataset kitti --DATA_PATH XXXXXXXXXX
```

Once the closed-set finished, we can move to evaluation (Fitting Known Classes and Model Inference). We provide a pretrained model for quick demonstration. You can simply run:

```
python evaluation.py --dataset kitti --DATA_PATH XXXXXXXXXX --OPEN_PATH XXXXXXXXXX --ckpt demo.pth
```

After the code is executed, we will get the performance of S2CA and a figure showing the OOD score distribution of known classes and unknown classes.

<p align="center">
  <img src="figures/pdf.png" width="700">
</p>


