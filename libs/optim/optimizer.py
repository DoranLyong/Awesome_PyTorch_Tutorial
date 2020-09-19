"""
code source: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/optim/optimizer.py
"""


import torch 
import torch.nn as nn 



def build_optimizer(
    model, 
    optim='adam',
    lr = 0.001,
    weight_decay = 0.0005, 
    momentum=0.9,
    ):

    if not isinstance(model, nn.Module):
        raise TypeError(
            'model given to build_optimizer must be an instance of nn.Module'
        )


    param_groups = model.parameters()
    

    if optim == 'adam':
        optimizer = torch.optim.Adam(
            params=param_groups,
            lr = lr, 
            weight_decay=weight_decay,
        )

    return optimizer



    