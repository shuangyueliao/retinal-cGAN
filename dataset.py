# from torch.utils import data

import os, sys
import numpy as np

import scipy.misc as m
from PIL import Image
from util import is_image_file, load_img
import torch
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

VOC2012_LOCAL_PATH = './VOC2012'


# TODO
# o shuffle support

class VOC2012Dataset(data_utils.Dataset):
    def __init__(self, root_path, set='train', img_size=256):
        self.root_path = root_path
        self.set = set
        self.img_size = img_size
        self.n_classes = 3
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

        assert self.set in ['train', 'val', 'trainval']
        self.files = []
        with open(self.root_path + '/ImageSets/Segmentation/' + self.set + '.txt', 'r') as f:
            for line in f:
                self.files.append(line.rstrip())

        self.files = sorted(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        base_name = self.files[index]
        img_file = self.root_path + '/JPEGImages/' + base_name
        gt_file = self.root_path + '/SegmentationClass/' + base_name
        realA = load_img(img_file)
        realA = self.transform(realA)
        if self.set == 'val':
            return base_name, realA
        else:
            realB = load_img(gt_file)
            realB = self.transform(realB)
            return realA, realB


if __name__ == '__main__':
    VOC2012_PATH = './VOC2012'
    ds_train = VOC2012Dataset(VOC2012_PATH, set='train')
    loader_train = data_utils.DataLoader(ds_train,
                                         batch_size=1,
                                         num_workers=1, shuffle=True)
    k = 0
    for i, (realA, realB) in enumerate(loader_train):
        print(realA, realB)
    print(k)
### EOF ###
