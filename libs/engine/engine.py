"""
[1] code source: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/engine/engine.py
[2] code source: https://github.com/dsgiitr/d2l-pytorch/blob/master/d2l/train.py
"""
import time 
import logging
import os.path as osp 
import datetime
from collections import OrderedDict

import numpy as np 
import torch 
import torch.nn.functional as F 



class Engine(object):
    def __init__(self, datamanager, use_gpu=True):

        self.datamanager = datamanager
        self.use_gpu = (torch.cuda.is_available() and use_gpu)

        self.model = None 
        self.optimizer = None 
        self.scheduler = None 

        self._models = OrderedDict() 
        self._optims = OrderedDict()
        self._scheds = OrderedDict() # lr_schedule 

    
    def register_model(self, name='model', model=None, optim=None, schedule=None):
        """
        https://www.daleseo.com/python-collections-ordered-dict/
        """
        self._models[name] = model 
        self._optims[name] = optim 
        self._scheds[name] = schedule

    
    def parse_data_for_train(self, data):
        imgs = [] # images 
        lbls = [] # labels

        

        return imgs, lbls



    def run(self, 
        save_dir='log',
        max_epoch=0, 
        start_epoch=0, 
        eval_freq=-1, 
        print_freq=10,
        test_only=False 
        ):
        
        if test_only:
            self.test()


    def train(self, print_freq=10, fixbase_epoch=0, open_layers=None):
        pass



    def test(self):
        pass 