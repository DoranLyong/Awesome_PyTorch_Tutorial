import torch 

from libs.data.fashion_mnist import * 
from libs.utils.figure import plt 
from libs.models.d2l_alexnet import d2l_AlexNet
from libs.optim import optimizer, lr_scheduler
from libs import engine





"""
1. DataLoader 
"""
train_iter, test_iter = load_data_fashion_mnist()

imgs = [] 
lbls = [] 

for idx, data in enumerate(test_iter):
    if(idx>=0 and idx<10):
        imgs.append(data[0])
        lbls.append(data[1])

    if (idx>=10):
        break 

show_fashion_mnist(imgs, get_fashion_mnist_labels(lbls))



"""
2. Build_model + optimizer + lr_scheduler 
"""
net = d2l_AlexNet()

# Model check 
X = torch.randn(size=(1,1,224,224))

for layer in net.model:
    X=layer(X)
    print(layer.__class__.__name__,'Output shape:\t',X.shape)


optimizer = optimizer.build_optimizer(
            model=net,
            optim='adam',
            lr=0.001,
            weight_decay=5e-04,
            )


scheduler = lr_scheduler.build_lr_scheduler(
            optimizer=optimizer,
            lr_scheduler='single_step',
            stepsize=20
            )


"""
3. Build_engine : 
"""
engine = engine.ImageSoftmaxEngine()



"""
4. Run training 
"""


"""
5. Testing
"""


