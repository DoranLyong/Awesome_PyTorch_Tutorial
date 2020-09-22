"""
[1] code source: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/engine/engine.py
[2] code source: https://github.com/dsgiitr/d2l-pytorch/blob/master/d2l/train.py
"""
import time 
import logging
import os.path as osp 
import datetime
from collections import OrderedDict

import coloredlogs
from tqdm import tqdm
import numpy as np 
import torch 
import torch.nn.functional as F 

from ..losses import DeepSupervision


coloredlogs.install(level="INFO", fmt="%(asctime)s %(filename)s %(levelname)s %(message)s")

class Engine(object):
    def __init__(self, datamanager, use_gpu=True):

        self.datamanager = datamanager
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.epoch = 0

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
        imgs = data['imgs'] # images 
        lbls = data['lbls'] # labels        

        return imgs, lbls

    def compute_loss(self, criterion, outputs, targets): 
        """
        Args: 
            * criterion: loss function 
            * outputs : model output 
            * targets : ground truth 
        """
        if isinstance(outputs, (tuple, list)):
            loss = DeepSupervision(criterion, outputs, targets)
        
        else: 
            loss =criterion(outputs, targets)
        
        return loss




    def run(self, 
        save_dir='log',
        max_epoch=0, 
        start_epoch=0, 
        eval_freq=-1, 
        print_freq=10,
        test_only=False 
        ):
        """
        A unified pipeline for training and evaluating a model.
        """

        time_start = time.time()
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        logging.info("=> Start training")

        # Training_loop 
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.train(
                print_freq=print_freq,               
            )

        


    def train(self, print_freq=10, fixbase_epoch=0, open_layers=None):
        
        train_l_sum = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        train_acc_sum = torch.tensor([0.0], dtype=torch.float32,device=self.device)
        n = 0 
        for X, y in tqdm(self.datamanager):
            loss_summary = self.forward_backward(X, y, mode='train')

            with torch.no_grad():
                y = y.long()
                train_l_sum += loss_summary['loss'].float()
#                train_acc_sum += (torch.sum((torch.argmax(loss_summary['outputs'], dim=1)==y))).float()
                n += y.shape[0]

                
        logging.info("Loss: %.4f " %(train_l_sum/n))



    def test(self):
        pass 


    def forward_backward(self, data):
        raise NotImplementedError