"""
[1] code source: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/engine/image/softmax.py
[2] code source: https://github.com/dsgiitr/d2l-pytorch/blob/master/d2l/train.py
"""
import logging

import coloredlogs
import torch 
import torch.nn as nn 

from ...data.fashion_mnist import *
from ..engine import Engine
from ...metrics import accuracy

coloredlogs.install(level="INFO", fmt="%(asctime)s %(filename)s %(levelname)s %(message)s")


class ImageNLLEngine(Engine):
    """
    CrossEntropyLoss == log_softmax + nll_loss 
    [ref] https://discuss.pytorch.org/t/is-log-softmax-nllloss-crossentropyloss/9352
    """

    def __init__(self, 
        datamanager, 
        model,
        optimizer, 
        scheduler=None, 
        use_gpu=True,
        label_smooth=True, 
        ):
        super(ImageNLLEngine, self).__init__(datamanager, use_gpu)

        self.model = model 
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model(name='model', model=model, optim=optimizer, schedule=scheduler)

        self.criterion = nn.CrossEntropyLoss()



    def forward_backward(self, X, y, mode='train'):
        imgs, lbls = X, y
#        show_fashion_mnist(imgs, get_fashion_mnist_labels(lbls))

#        print("input shape: ", imgs.shape )
#        print("label shape: ", lbls.shape)

        if self.use_gpu:
            imgs = imgs.cuda() 
            lbls = lbls.cuda() 
            self.model = self.model.cuda()

        if mode =='train':
            self.model.train()  # Switch to training mode 

        else: 
            self.model.eval()

        outputs = self.model(imgs)
#        loss = self.compute_loss(self.criterion, outputs, lbls)
        loss = self.criterion(outputs, lbls)

        
        # _Start: backpropagation & update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

#        loss_summary = {
#            'loss' : loss.item(), 
#            'acc' : accuracy(outputs, lbls)
#        }


        loss_summary = {'loss':loss, 'outputs':outputs}

        return loss_summary


        
