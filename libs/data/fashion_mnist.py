"""
[ref]: https://github.com/dsgiitr/d2l-pytorch/blob/master/d2l/data/fashion_mnist.py

* Download Fashion-MNIST dataset 
* Get labels 
* Show the dataset 
"""
import sys 
sys.path.insert(0, '..')
import os 
import os.path as osp
import logging 

import coloredlogs
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from ..utils.figure import test, plt

coloredlogs.install(level="INFO", fmt="%(asctime)s %(filename)s %(levelname)s %(message)s")


def load_data_fashion_mnist(batch_size=1, resize=None, root=osp.join(osp.dirname(__file__), 'datasets','fashion_mnist')):
    """
    DownLoad the Fashion-MNIST dataset and then load into memory. 
    """
    root = os.path.expanduser(root)  # convert to absolute path 
    logging.info("the ROOT path of Fashion-MNIST: {}".format(root))
    
    transformer = []
    
    if resize:
        transformer += [transforms.Resize(resize)]
    transformer += [transforms.ToTensor()]
    transformer = transforms.Compose(transformer)

    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, transform=transformer, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, transform=transformer, download=True)

    num_workers = 0 if sys.platform.startswith('win32') else 4


    train_iter = DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers)
    test_iter = DataLoader(mnist_test, batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


def get_fashion_mnist_labels(labels):
    """
    Get text labels for Fashion-MNIST.
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images,labels):
    """
    Plot Fashion-MNIST images with labels.
    """
    
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    print(len(images))
    
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
        
    plt.show()




    
    
    
    

    


