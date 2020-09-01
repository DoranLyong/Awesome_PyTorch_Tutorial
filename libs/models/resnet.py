''' ResNet in Pytorch 
[1] https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnet18
[2] https://arxiv.org/abs/1512.03385
[3] https://github.com/KaiyangZhou/pytorch-cifar/blob/master/models/resnet.py
[4] https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
'''

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18']


model_urls = {
    'resnet18':
    'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34':
    'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50':
    'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101':
    'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152':
    'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d':
    'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d':
    'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


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
    """
    ResNet v1.5
    """
    expansion = 4 

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1,
                    base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d 

        width = int(out_channels * (base_width / 64.)) * groups
        """
        Both self.conv2 and self.downsample layers downsample the input when stride != 1
        """
        self.conv1 = conv1x1(in_channels, width)
        self.bn1   = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2   = norm_layer(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion)
        self.bn3   = norm_layer(out_channels * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x 

        # === bottleneck === # 
        out = self.conv1(x)
        out = self.bn1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None: 
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, 
                groups=1, width_per_group=64, replace_stride_with_dilation=None,
                norm_layer=None):
    
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channels = 64
        self.dilation = 1 
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None ", "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group 
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = norm_layer(self.in_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1  = self._make_layer(block=block, planes=64, num_blocks=layers[0] )
        self.layer2  = self._make_layer(block=block, planes=128, num_blocks=layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0]
                                        )
        self.layer3  = self._make_layer(block=block, planes=256, num_blocks=layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1]
                                        )
        self.layer4  = self._make_layer(block=block, planes=512, num_blocks=layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2]
                                        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)



    def _make_layer(self, block, planes, num_blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None 
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride 
            stride = 1 

        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.in_channels = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpoll(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


def init_pretrained_weights( model_url):
    """Initializes model with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    [1] https://pytorch.org/docs/stable/model_zoo.html

    Args: 
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return model_zoo.load_url(url= model_url ,progress=True)
    
### ==== #### 

def _resnet(arch, block, layers, pretrained, **kwargs):
    
    model = ResNet(block=block, layers= layers, **kwargs)
    
    if pretrained: 
        pretrain_dict = init_pretrained_weights( model_url = model_urls[arch])

    model_dict = model.state_dict()

    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    return model



def resnet18(num_classes, pretrained=True, **kwargs):

    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, **kwargs)




def test():
    net = resnet18(num_classes=10)
    y = net(torch.randn(1,3,32,32))
    print(y.size())

test()
    