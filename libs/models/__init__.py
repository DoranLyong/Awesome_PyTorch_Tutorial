"""
code source : https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/models/__init__.py
"""
import logging

import coloredlogs
import torch

from .d2l_alexnet import *


coloredlogs.install(level="INFO", fmt="%(asctime)s %(filename)s %(levelname)s %(message)s")

__model_factory = {
        # image classification models 
    'd2l_alexnet' : d2l_AlexNet


}


def show_avai_models(): 
    """
    Show available models. 

    Example:
            >> models.show_avai_models()
    """
    
    logging.info("AVAI_model list: {}".format(list(__model_factory.keys())))



def build_model(
    model_name,
    num_classes,
    loss='softmax', 
    pretrained=True,
    use_gpu=True,

    ):

    avai_models = list(__model_factory.keys())

    if model_name not in avai_models:
        raise KeyError('Unknown model: {}. It must be one of {}'.format(model_name, avai_models))


    logging.info("Building_model with : {}".format(avai_models))



    built_model = __model_factory[model_name]( num_classes=10, loss=loss)


    return built_model
    