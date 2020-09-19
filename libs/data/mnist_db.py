import logging 
import os 
import os.path as osp 
from glob import glob 

import numpy as np 
from PIL import Image 
import coloredlogs
from torch.utils.data import Dataset 
from torchvision import transforms 

from utils.config import cfg

coloredlogs.install(level="INFO", fmt="%(asctime)s %(filename)s %(levelname)s %(message)s")



def get_label(path):
    lbl_name = path.split('/')[-2]    
    return int(lbl_name)

def get_classes(paths):
    return np.unique( os.listdir(paths))



class MNIST_DB(Dataset): 
    """ MNIST_png database """ 

    def __init__(self, db_name ,root_dir=None, transform=None):
        super(MNIST_DB, self).__init__()
        self.db_name = db_name 
        self.root_dir = root_dir if root_dir else osp.join(cfg.DATASET_DIR, "mnist_png")
        logging.info("the ROOT path of MNIST: {}".format(self.root_dir))  
        self.transform = transform 
        
        if db_name == "mnist_train":
            self.data_dir = osp.join(self.root_dir, "training")           
        else: 
            self.data_dir = osp.join(self.root_dir, "testing")

        self.list_datapath = glob(osp.join(self.data_dir, '*','*.png')) 
        self.classes = get_classes(self.data_dir)
                              

    def __len__(self):
        return len(self.list_datapath) 

    def __getitem__(self, idx):        
        path = self.list_datapath[idx]

        # Read Image 
        image_pil = Image.open(path).convert("L") # for gray_scale

        # Get Label 
        label = get_label(path)


        if self.transform: 
            image = self.transform(image_pil)
        else: 
            image = transforms.ToTensor()(image_pil)

        return image, label



        
