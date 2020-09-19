"""
code source: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/optim/lr_scheduler.py
"""
import torch 


def build_lr_scheduler(
    optimizer, 
    lr_scheduler='single_step',
    stepsize=1,
    gamma=0.1,

    ):


    if lr_scheduler == 'single_step':

        if not isinstance(stepsize, int):
            raise TypeError(
                'For single_step lr_scheduler, stepsize must '
                'be an integer, but got {}'.format(type(stepsize))
                )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma= gamma)


    return scheduler

