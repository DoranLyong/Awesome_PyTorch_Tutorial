'''ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnet18
[3] https://github.com/KaiyangZhou/pytorch-cifar/blob/master/models/resnet.py
'''
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class BasicBlock(nn.Module):
    expansion = 1 

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()