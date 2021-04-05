'''
Author: Kunchang Li
Date: 2021-04-05 15:47:18
LastEditors: Kunchang Li
LastEditTime: 2021-04-05 18:16:21
'''

import os
import random

from .dataset import Dataset
from paddleseg.utils import seg_env
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


@manager.DATASETS.add_component
class RemoteSensing(Dataset):
    """
    Args:
        transforms (list): Transforms for image.
        train_dataset_root (str): The training dataset directory. Default: None
        test_dataset_root (str): The training dataset directory. Default: None
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """
    NUM_CLASSES = 7

    def __init__(self,
                 train_dataset_root=None,
                 test_dataset_root=None,
                 negetive_ratio=0,
                 positive_train_dataset_list=None,
                 negetive_train_dataset_list=None,
                 transforms=None,
                 mode='train',
                 edge=False):
        self.train_dataset_root = train_dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255
        self.edge = edge

        if mode not in ['train', 'val', 'test']:
            raise ValueError(
                "`mode` should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if mode == 'train':
            file_path = os.path.join(self.train_dataset_root, 'train_list.txt')
        elif mode == 'val':
            file_path = os.path.join(self.train_dataset_root, 'val_list.txt')
        else:
            file_path = os.path.join(self.test_dataset_root, 'test_list.txt')

        
        if mode == 'train' and   negetive_ratio != 0:
            positive_file_path = os.path.join(self.train_dataset_root, positive_train_dataset_list)
            negetive_file_path = os.path.join(self.train_dataset_root, negetive_train_dataset_list)
            with open(positive_file_path, 'r') as f:
                lines = f.readlines()
                positive_lines = [line for line in lines]
                positive_length = len(positive_lines)
            with open(negetive_file_path, 'r') as f:
                lines = f.readlines()
                negetive_lines = [line for line in lines]
                negetive_length = len(negetive_lines)
            if int(positive_length * negetive_ratio) < negetive_length:
                negetive_length = int(positive_length * negetive_ratio)
            sample_lines = positive_lines + random.sample(negetive_lines, int(negetive_length))
            for line in sample_lines:
                items = line.strip().split()
                image_path = os.path.join(self.train_dataset_root, items[0])
                grt_path = os.path.join(self.train_dataset_root, items[1])
                self.file_list.append([image_path, grt_path])
            print(f"{positive_length} positive data from :", negetive_train_dataset_list)
            print(f"Add {negetive_length} negetive data from :", negetive_train_dataset_list)
            print(f"Total data for {mode} : {len(self.file_list)}")
        else:
            with open(file_path, 'r') as f:
                for line in f:
                    items = line.strip().split()
                    if len(items) != 2:
                        if mode == 'train' or mode == 'val':
                            raise Exception(
                                "File list format incorrect! It should be"
                                " image_name label_name\\n")
                        image_path = os.path.join(self.test_dataset_root, items[0])
                        grt_path = None
                    else:
                        image_path = os.path.join(self.train_dataset_root, items[0])
                        grt_path = os.path.join(self.train_dataset_root, items[1])
                    self.file_list.append([image_path, grt_path])
            print(f"Total data for {mode} : {len(self.file_list)}")
        
