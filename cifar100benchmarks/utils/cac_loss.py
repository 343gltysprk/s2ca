import copy
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvt
from torch import Tensor

class CACLoss(nn.Module):
    """
    Class Anchor Clustering Loss from the paper
    *Class Anchor Clustering: a Distance-based Loss for Training Open Set Classifiers*
    :see Paper: `WACV 2022 <https://arxiv.org/abs/2004.02434>`_

    """

    def __init__(self, num_classes=100, alpha=10,lbda=0.3):
        super(CACLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.lbda = lbda
        self.anchors = nn.Parameter(torch.zeros(self.num_classes, self.num_classes).double(), requires_grad = False)

    def distance_classifier(self, x):
    
        #Calculates euclidean distance from x to each class anchor
        #Returns n x m array of distance from input of batch_size n to anchors of size m

        n = x.size(0)
        m = self.num_classes
        d = self.num_classes
        
        x = x.unsqueeze(1).expand(n, m, d).double()
        anchors = self.anchors.unsqueeze(0).expand(n, m, d)
        dists = torch.norm(x-anchors, 2, 2)
        return dists
        
    def loss_fn(self, distances, gt):
        '''Returns CAC loss, as well as the Anchor and Tuplet loss components separately for visualisation.'''
        true = torch.gather(distances, 1, gt.view(-1, 1)).view(-1)
        non_gt = torch.Tensor([[i for i in range(self.num_classes) if gt[x] != i] for x in range(len(distances))]).long().cuda()
        others = torch.gather(distances, 1, non_gt)

        anchor = torch.mean(true)

        tuplet = torch.exp(-others+true.unsqueeze(1))
        tuplet = torch.mean(torch.log(1+torch.sum(tuplet, dim = 1)))

        total = self.lbda*anchor + tuplet

        return total, anchor, tuplet
        
