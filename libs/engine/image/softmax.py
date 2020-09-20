"""
[1] code source: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/engine/image/softmax.py
[2] code source: https://github.com/dsgiitr/d2l-pytorch/blob/master/d2l/train.py
"""

from ..engine import Engine



class ImageNLLEngine(Engine):
    """
    Negative Log-likelihood loss 
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

        #self.criterion = 



    def forward_backward(self, data):
        imgs, lbls = self.parse_data_for_train(data)