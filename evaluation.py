from __future__ import print_function

from torchvision.io import read_image
import os
import torch
import torch.nn as nn
import torchvision
import pandas as pd
import torchvision.transforms as tvt
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import pytorch_ood
from dataset import ModelNetDataLoader
import data_helper as helper
from networks.WRN40 import WideResNet
from networks.voxnet import VoxNet
from networks.lightnet import light
from networks.dgcnn import DGCNN
from networks.pct import PointTransformerCls
from pytorch_ood.utils import OODMetrics, ToRGB, ToUnknown
import numpy as np
from pytorch_ood.detector import (
    ODIN,
    EnergyBased,
    Entropy,
    KLMatching,
    MaxLogit,
    MaxSoftmax,
    ViM,
)

import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn

from torchvision import transforms, datasets
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.resnet_big import SupConResNet, LinearClassifier, LinearC, SupConGeneric

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass



def set_loader(opt):
    DATA_PATH = opt.DATA_PATH
    train_dataset = ModelNetDataLoader(root=DATA_PATH, npoint=1024, uniform=False, split='train', normal_channel=False)
    closed = ModelNetDataLoader(root=DATA_PATH, npoint=1024, uniform=False, split='test', normal_channel=False)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    closed_loader = torch.utils.data.DataLoader(
        closed, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader, closed_loader




def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='kitti',
                        choices=['nu','kitti'], help='name of dataset')
    parser.add_argument('--DATA_PATH', type=str, default=r'/data', help='dataset location')
    parser.add_argument('--OPEN_PATH', type=str, default=r'/data', help='unknown classes')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default='./datasets/', help='path to custom dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='demo.pth' ,help='path to pre-trained model')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')

    opt = parser.parse_args()

    # set the path according to the environment

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'nu':
        opt.n_cls = 19
    elif opt.dataset == 'kitti':
        opt.n_cls = 4
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_model(opt):
    model = DGCNN(opt,output_channels=opt.n_cls)
    criterion = torch.nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(opt.ckpt))
    
    model.cuda()

    return model, criterion



def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()
    loss_fn = pytorch_ood.loss.EntropicOpenSetLoss().cuda()

    batch_time = AverageMeter()
    losses = AverageMeter()
    
    correct = 0
    total = 0

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]
            total += bsz

            # forward
            features = images
            output = model.predict(features.detach())
            loss = loss_fn(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            correct += accuracy(output, labels)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses))

    print(' * Accuracy: ',correct/total)
    return losses.avg, correct/total





def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, closed_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)
    
    model.eval()
    

    # eval for one epoch
    loss, val_acc = validate(closed_loader, model, criterion, opt)


    print('best accuracy: {:.4f}'.format(val_acc))
    
    
    std = None
    
    OPEN_PATH = opt.OPEN_PATH
    ood_data = ModelNetDataLoader(root = OPEN_PATH,npoint=1024, uniform=False, split='test', normal_channel=False)
    
    classes = [ torch.tensor([0]) , torch.tensor([1]), torch.tensor([2]) ,torch.tensor([3])]
    indices = []
    for i in range(4):
        idc = []
        for j in range(ood_data.__len__()):
            if (ood_data.__getitem__(j)[1]==i):
                idc.append(j)
        indices.append(idc)
    closed = ModelNetDataLoader(root=opt.DATA_PATH, npoint=1024, uniform=False, split='test', normal_channel=False)
    ood_data = ModelNetDataLoader(root = OPEN_PATH,npoint=1024, uniform=False, split='test', normal_channel=False, open_set = True)
    ood_loader = torch.utils.data.DataLoader(closed+ood_data, batch_size=32, shuffle=False,num_workers=4, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from   sklearn.decomposition import PCA
    from   sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding
    from   sklearn.preprocessing import StandardScaler
    import seaborn as sns
    
    
    
    detectors = {}
    detectors["ViM"] = ViM(model.features, d=64, w=model.linear3.weight, b=model.linear3.bias)
    
    print(f"> Fitting Known Classes")

    for name, detector in detectors.items():
        detector.fit(train_loader, device=device)


    results = []

    with torch.no_grad():
        for detector_name, detector in detectors.items():
            print(f"> Evaluating Open Set Performance")

            metrics = OODMetrics()
            for x, y in ood_loader:
                metrics.update(detector(x.to(device)), y.to(device))

            r = {"Detector": detector_name}
            r.update(metrics.compute())
            results.append(r)

    # calculate mean scores over all datasets, use percent
    df = pd.DataFrame(results)
    mean_scores = df.groupby("Detector").mean() * 100
    print(mean_scores.sort_values("AUROC").to_csv(float_format="%.2f"))


    for detector_name, detector in detectors.items():
        scores = []
        labels = []
        known = []
        for x, y in ood_loader:
            a = list(detector(x.to(device)).detach().cpu().numpy())
            b = list(y.detach().cpu().numpy())
            scores += a
            labels += b
        for i in labels:
            if (i<0):
                known.append('Unknown')
            else:
                known.append('Known')
        data = np.array(scores)
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        df = pd.DataFrame({'Normalized Score': data, 'Category': known})
        ax = sns.displot(data = df, x='Normalized Score', hue='Category', kde=True, stat="probability",common_norm=False,aspect=4/3)
        sns.move_legend(ax, "upper right")
        plt.xlabel('Normalized Score')
        plt.ylabel('Probability Density')
        plt.savefig('Probability_Density.png',dpi=1200)

    
if __name__ == '__main__':
    main()
