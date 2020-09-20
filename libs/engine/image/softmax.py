"""
[1] code source: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/engine/image/softmax.py
[2] code source: https://github.com/dsgiitr/d2l-pytorch/blob/master/d2l/train.py
"""

from ..engine import Engine



class ImageSoftmaxEngine(Engine):
    """
    Softmax-loss engine 
    """

    def __init__(self, 
        datamanager, 
        model,
        optimizer, 
        scheduler=None, 
        use_gpu=True,
        label_smooth=True, 
        ):
        super(ImageSoftmaxEngine, self).__init__(datamanager, use_gpu)

        self.model = model 
        self.optimizer = optimizer
        self.scheduler =scheduler
        self.register_model(name='model', model=model, optim=optimizer, schedule=scheduler)

        self.criterion = 