import torch
import torch.nn as nn
from collections import OrderedDict


class VoxNet(nn.Module):
    def __init__(self, n_classes=15, input_shape=(32, 32, 32), alpha=10, lbda=0.3):
        super(VoxNet, self).__init__()
        self.num_classes = n_classes
        self.alpha = alpha
        self.lbda = lbda
        
        self.anchors = nn.Parameter(torch.zeros(self.num_classes, self.num_classes).double(), requires_grad = False)
        self.input_shape = input_shape
        self.feat = torch.nn.Sequential(OrderedDict([
            ('conv3d_1', torch.nn.Conv3d(in_channels=1,
                                         out_channels=32, kernel_size=5, stride=2)),
            ('relu1', torch.nn.ReLU()),
            ('drop1', torch.nn.Dropout(p=0.2)),
            ('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
            ('relu2', torch.nn.ReLU()),
            ('pool2', torch.nn.MaxPool3d(2)),
            ('drop2', torch.nn.Dropout(p=0.3))
        ]))
        x = self.feat(torch.autograd.Variable(torch.rand((1, 1) + input_shape)))
        dim_feat = 1
        for n in x.size()[1:]:
            dim_feat *= n

        self.fc1 = torch.nn.Linear(dim_feat, 128)
        self.relu1 = torch.nn.ReLU()
        self.drop3 = torch.nn.Dropout(p=0.4)
        self.fc2 = torch.nn.Linear(128, self.num_classes)
        
        self.head = nn.Sequential(
                nn.Linear(dim_feat, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128)
            )
        
            
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


    def set_anchors(self, means):
        self.anchors = nn.Parameter(means.double(), requires_grad = False)
        self.cuda()
        
        
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

    def forward(self, x):
        x = x.float()
        x = self.feat(x)
        x = x.view(x.size(0), -1)
        aux = self.head(x)
        aux = torch.nn.functional.normalize(aux)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop3(x)
        x = self.fc2(x)
        dist  = self.distance_classifier(x)
        return x, dist, aux
        
    def predict(self, x):
        x = x.float()
        x = self.feat(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop3(x)
        x = self.fc2(x)
        return x
        

    def features(self, x):
        x = x.float()
        x = self.feat(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        #x = self.drop3(x)
        return x

if __name__ == "__main__":
    voxnet = VoxNet()
    data = torch.rand([256, 1, 32, 32, 32])
    voxnet(data)