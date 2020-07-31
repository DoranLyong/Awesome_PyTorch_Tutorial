import argparse 
import logging 
import os 
import os.path as osp 
import random 
import time 

import numpy as np 
import matplotlib.pyplot as plt
import coloredlogs
import torch 
import torch.nn.functional as F
import torch.optim as optim 
from torch.optim.lr_scheduler import ReduceLROnPlateau   # learning_rate scheduler 
from torch.utils.data import DataLoader
from torchvision import transforms 

import _init_paths
from utils.config import cfg 
from datasets.mnist_db import MNIST_DB
from models.mnist_Net import mnistNet


coloredlogs.install(level="INFO", fmt="%(asctime)s %(filename)s %(levelname)s %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(description="Train a network")
    parser.add_argument(
        "--gpu", default=-1, type=int, help="GPU device id to use. Default: -1, means using CPU "
    )
    parser.add_argument(
        "--dataset", default="mnist_train", help="Dataset to train on. Default: mnist_train"
    )
    parser.add_argument(
        "--epochs", default=20, type=int, help="Number of epochs to train. Default: 20"
    )
    parser.add_argument(
        "--batch_size", default=64, type=int, help="Number of batch_size to train. Default: 64"
    )
    parser.add_argument(
        "--seed", default=1, type=int, help="Seed number. Default: 1"
    )
    return parser.parse_args()



def data_visualize(img, label):
    fig, ax = plt.subplots(figsize=(16,9))
    ax.imshow(img)
    plt.axis("off")
    plt.title(label)
    plt.show()
    plt.close(fig)


def vis_dataloader(data, target): 
     
    a = torch.squeeze(data[0])
    img = a.numpy() 
    label = target[0].numpy()    
    data_visualize(img, label)

          

if __name__ == "__main__":
    args = parse_args()

    output_dir = osp.join(cfg.DATA_DIR, "trained_model", args.dataset)
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    assert args.dataset in ["mnist_train", "mnist_test"], "Unknown dataset: %s" % args.dataset



    torch.manual_seed(args.seed)
    lr       = cfg.TRAIN.LEARNING_RATE
    momentum = cfg.TRAIN.MOMENTUM
    batch_size = args.batch_size 
    epochs = args.epochs    
    no_cuda = False 
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory':True} if use_cuda else {}  


    # DataLoader
    trans_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.406],
                    std=[0.225]               
                )])

    trans_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.406],
                    std=[0.225]               
                )])


    train_loader = DataLoader( MNIST_DB(args.dataset, root_dir=None, transform=trans_train), 
                            batch_size=batch_size,
                            shuffle = True,
                            **kwargs,
    )   
    test_loader = DataLoader( MNIST_DB("mnist_test", root_dir=None, transform=trans_test), 
                            batch_size=batch_size,
                            shuffle = False,
                            **kwargs,
    )   


    
    # Initialize model 
    net = mnistNet().to(device)

    # Initialize optimizer 
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, #receeding from 0.1 -> lr  
                                  patience=5, verbose=True,         
    )

    # Training settings 

    log_interval = 100

    for epoch in range(1, epochs + 1): 
        #_Train 
        net.train()
        
        for batch_idx, (data,target) in enumerate(train_loader):
#            vis_dataloader(*(data,target))
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step() 

            if batch_idx % log_interval == 0 : 
                logging.info("Train Epoch: {}[{}/{} ({:.0f}%)] \tLoss: {:.6f}".format(
                    epoch, batch_idx*len(data), len(train_loader.dataset), 
                    100 * batch_idx / len(train_loader), loss.item()
                ))

        #_Validation 
        net.eval() 

        test_loss = 0 
        correct = 0 

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = net(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        accuracy = 100. * correct/len(test_loader.dataset)
        
        logging.info("\nTest set: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
              test_loss, correct, len(test_loader.dataset),accuracy ))

        #_Learning Rate Scheduler 
        scheduler.step(accuracy, epoch)  # message when the 'lr' is reduced 




            
    
    
