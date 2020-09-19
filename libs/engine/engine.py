"""
[1] code source: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/engine/engine.py
[2] code source: https://github.com/dsgiitr/d2l-pytorch/blob/master/d2l/train.py
"""
import time 
import logging
import os.path as osp 
import datetime

import numpy as np 
import torch 
import torch.nn.functional as F 



class Engine(object):
    def __init__(self, use_gpu=True):

        self.use_gpu = (torch.cuda.is_available() and use_gpu)

        self.model = None 
        self.optimizer = None 
        self.scheduler = None 


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