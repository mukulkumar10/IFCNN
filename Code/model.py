import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import skimage.feature as ski
import numpy as np



# My Convolution Block
class ConvBlock(nn.Module):
    def __init__(self, inplane, outplane):
        super(ConvBlock, self).__init__()
        self.padding = (1, 1, 1, 1)
        self.conv = nn.Conv2d(inplane, outplane, kernel_size=3, padding=0, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = F.pad(x, self.padding, 'replicate')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

    
class IFCNN(nn.Module):
    def __init__(self, resnet, fuse_scheme=0):
        super(IFCNN, self).__init__()
        self.fuse_scheme = fuse_scheme # MAX, MEAN, SUM
        self.conv1 = ConvBlock(1,64)
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1, bias=True)

        # Initialize parameters for other parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        # Initialize conv1 with the pretrained resnet101 and freeze its parameters
        # for p in resnet.parameters():
        #     p.requires_grad = False
        # self.conv1 = resnet.conv1
        # self.conv1.stride = 1
        # self.conv1.padding = (0, 0)

    def tensor_max(self, tensors):
        # max_tensor = None
        
        # for i, tensor in enumerate(tensors):
        #     if i == 0:
        #         max_tensor = tensor
        #     else:
        #         max_tensor = torch.max(max_tensor, tensor)
        max_tensor=None
        for _,i in enumerate(range(64)):
              tensor1=tensors[0][0][i]
              tensor2=tensors[1][0][i]
              
  
              arr1=tensor1.cpu().detach().numpy().astype(np.uint8)
              arr2=tensor2.cpu().detach().numpy().astype(np.uint8)
              g1=ski.greycomatrix(arr1,[1],[0],levels=256)
              g2=ski.greycomatrix(arr2,[1],[0],levels=256)
              c1=ski.greycoprops(g1,'contrast')[0][0]
              e1=ski.greycoprops(g1,'energy')[0][0]
              c2=ski.greycoprops(g2,'contrast')[0][0]
              e2=ski.greycoprops(g2,'energy')[0][0]
              if(_==0):
                if(e1>e2):
                  max_tensor=torch.unsqueeze(tensors[0][0][i],0)
                else:
                  max_tensor=torch.unsqueeze(tensors[1][0][i],0)
              else:
                if(e1>e2):
                  max_tensor=torch.cat((max_tensor,torch.unsqueeze(tensors[0][0][i],0)),0)
                else:
                  max_tensor=torch.cat((max_tensor,torch.unsqueeze(tensors[1][0][i],0)),0)
                
        max_tensor=torch.unsqueeze(max_tensor,0)      
        

        return max_tensor

    def tensor_sum(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        return sum_tensor

    def tensor_mean(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        mean_tensor = sum_tensor / len(tensors)
        return mean_tensor

    def operate(self, operator, tensors):
        out_tensors = []
        for tensor in tensors:
            out_tensor = operator(tensor)
            out_tensors.append(out_tensor)
        return out_tensors

    def tensor_padding(self, tensors, padding=(1, 1, 1, 1), mode='constant', value=0):
        out_tensors = []
        for tensor in tensors:
            out_tensor = F.pad(tensor, padding, mode=mode, value=value)
            out_tensors.append(out_tensor)
        return out_tensors

    def forward(self, *tensors):
        # Feature extraction
        #outs = self.tensor_padding(tensors=tensors, padding=(1, 1, 1, 1), mode='constant')
        outs = self.operate(self.conv1, tensors)
        outs = self.operate(self.conv2, outs)
        
        # Feature fusion
        if self.fuse_scheme == 0: # MAX
            out = self.tensor_max(outs)
        elif self.fuse_scheme == 1: # SUM
            out = self.tensor_sum(outs)
        elif self.fuse_scheme == 2: # MEAN
            out = self.tensor_mean(outs)
        else: # Default: MAX
            out = self.tensor_max(outs)
        
        # Feature reconstruction
        out = self.conv3(out)
        out = self.conv4(out)
        return out


def myIFCNN(fuse_scheme=0):
    # pretrained resnet101
    resnet = models.resnet101(pretrained=True)
    # our model
    model = IFCNN(resnet, fuse_scheme=fuse_scheme)
    return model
