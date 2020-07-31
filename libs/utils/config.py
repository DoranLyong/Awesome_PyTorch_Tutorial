import os 
import os.path as osp 

import numpy as np 
import yaml 
from easydict import EasyDict 

cfg = EasyDict()  # {cfg.TRAIN:{}, cfg.TEST:{}, DATA_DIR: ,  }


####################
# Training options #
####################

cfg.TRAIN = EasyDict()

# Learning rate 
cfg.TRAIN.LEARNING_RATE = 0.001 
# Momentum 
cfg.TRAIN.MOMENTUM = 0.5
# Minibatch size 
cfg.TRAIN.BATCH_SIZE = 64





########
# MISC #
########

# Root directory of project 
cfg.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))

# Data directory 
cfg.DATA_DIR = osp.join(cfg.ROOT_DIR, "data")
cfg.DATASET_DIR = osp.join(cfg.ROOT_DIR, "data","dataset")

# Default GPU device id 
cfg.GPU_ID = 0 
