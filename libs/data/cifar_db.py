import logging 
import os 
import os.path as osp 
from glob import glob 

import numpy as np 
from PIL import Image 
import coloredlogs
from  torch.utils.data import Dataset
from torchvision import transforms 

from utils.config import cfg 

coloredlogs.install(level="INFO", fmt="%(asctime)s %(filename)s %(levelname)s %(message)s")



def get_name(path):
    lbl_name = osp.basename(path).split('_')[-1].replace('.png', '')
    return lbl_name


def get_classes(paths):
    label_names = [get_name(path) for path in paths ]
    classes = np.unique(label_names)
    return classes



class cifar_DB(Dataset):
    """ cifar database """

    def __init__(self, db_name, root_dir=None, transform=None):
        super(cifar_DB, self).__init__()
        self.db_name = db_name 
        self.root_dir = root_dir if root_dir else osp.join(cfg.DATASET_DIR , "cifar")
        logging.info("the ROOT path of cifar: {}".format(self.root_dir))
        self.transform = transform 

        if db_name == 'cifar_train':
            self.data_dir = osp.join(self.root_dir, "train")
        else:
            self.data_dir = osp.join(self.root_dir, 'test')

        logging.info("the path of {0} dir: {1}".format(db_name, self.data_dir))
        
        self.list_datapath = glob(osp.join(self.data_dir,'*.png'))
        self.classes = get_classes(self.list_datapath)
        logging.info("classes = {}".format(self.classes))
        
    def __len__(self):
        return len(self.list_datapath)

    def __getitem__(self, idx):
        path = self.list_datapath[idx]

        # Read Image 
#        image_pil = Image.open(path).convert("L")   # for gray_scale
        image_pil = Image.open(path)

        # Get Label 
        label = self.get_label(path)

        if self.transform:
            image = self.transform(image_pil)
        else:
            image = transforms.ToTensor()(image_pil)
        
        return image, label 

    def get_label(self, path):
        lbl_name = osp.basename(path).split('_')[-1].replace('.png', '')
        label = np.argmax(self.classes == lbl_name)
        return label