''' ResNet in Pytorch 
[1] https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnet18
[2] https://arxiv.org/abs/1512.03385
[3] https://github.com/KaiyangZhou/pytorch-cifar/blob/master/models/resnet.py
[4] https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F


__all__ = ['ResNet', 'resnet18']



def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """
    3x3 convolution with padding 
    """
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                    padding=dilation, groups=groups, bias=False, dilation=dilation
                    )                        

def conv1x1(in_channels, out_channels, stride=1):
    """
    1x1 convolution
    """
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                    bias=False
                    )


class BasicBlock(nn.Module):
    expansion = 1 

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1, 
                base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()

        if norm_layer is None: 
            norm_layer = nn.BatchNorm2d
        
        if groups != 1 or base_width != 64: 
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")


        '''
        Both self.conv1 and self.downsample layers downsample the input when stride != 1
        '''
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1   = norm_layer(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2   = norm_layer(out_channels)
        
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        identity = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out 







class Bottleneck(nn.Module):
    expansion = 4 



