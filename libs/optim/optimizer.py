"""
code source: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/optim/optimizer.py
"""

import torch 
import torch.nn as nn 


#_Available options
AVAI_OPTIMS = ['adam',  'sgd', ] 


def build_optimizer(
    model, 
    optim='adam',
    lr = 0.001,
    weight_decay = 0.0005, 
    momentum=0.9,
    sgd_dampening=0,
    sgd_nesterov=False,
    ):

    if not isinstance(model, nn.Module):
        raise TypeError(
            'model given to build_optimizer must be an instance of nn.Module'
        )

    param_groups = model.parameters()  # get learnable parameters 




    if optim not in AVAI_OPTIMS:
        raise ValueError(
            'Unsupported optim: {}. It must be one of {}'.format( optim, AVAI_OPTIMS )
        )
    

    if optim == 'adam':
        """
        https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
        """
        optimizer = torch.optim.Adam(
            params=param_groups,
            lr = lr, 
            weight_decay=weight_decay,
        )


    elif optim == 'sgd':
        """
        https://pytorch.org/docs/stable/optim.html#torch.optim.SGD
        """
        optimizer = torch.optim.SGD(
            params=param_groups, 
            lr=lr, 
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
            )

    return optimizer



    