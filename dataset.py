#from torch.utils import data

import os,sys
import numpy as np

import scipy.misc as m
from PIL import Image
from util import is_image_file, load_img
import torch
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
VOC2012_LOCAL_PATH='./VOC2012'

# TODO
# o shuffle support

class VOC2012Dataset(data_utils.Dataset):
    MEAN = np.array([104.00699, 116.66877, 122.67892])

    def __init__(self, root_path, set='train', img_size=256):
        self.root_path = root_path
        self.set       = set
        self.img_size  = img_size
        self.n_classes = 3
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

        assert self.set in ['train', 'val', 'trainval']

        #self.mean = np.array([104.00699, 116.66877, 122.67892])

        self.files = []
        with open (self.root_path + '/ImageSets/Segmentation/' + self.set + '.txt', 'r') as f:
            for line in f:
                self.files.append(line.rstrip())

        self.files = sorted(self.files)


    def __len__(self):
        return len(self.files)


    def _get_image(self, path):
        img   = m.imread(path)
        npimg = np.array(img, dtype=np.uint8)

        # RGB => BGR
        npimg = npimg[:, :, ::-1] # make a copy of the same list in reverse order:
        npimg = npimg.astype(np.float64)
        npimg -= VOC2012Dataset.MEAN

        npimg = m.imresize(npimg, (self.img_size, self.img_size))

        npimg = npimg.astype(float) / 255.0

        npimg = npimg.transpose(2,0,1) # (3, 256, 256)

        return torch.from_numpy(npimg).float()


    def _get_pascal_labels(self):
        return np.asarray([[0, 0, 0], [255, 0, 0], [0, 255, 0]])


    def _encode_segmap(self, npgt):
        # npgt contains 255
        npgt = npgt.astype(int)

        npgt2 = np.zeros((npgt.shape[0], npgt.shape[1]), dtype=np.int16)
        for i, label in enumerate(self._get_pascal_labels()):
            npgt2[np.where(np.all(npgt == label, axis=-1))[:2]] = i

        npgt2 = npgt2.astype(int)

        return npgt2


    def _get_gt(self, path):
        gt = m.imread(path)
        npgt = np.array(gt, dtype=np.int32)

        npgt = self._encode_segmap(npgt)
        classes = np.unique(npgt)
        npgt = npgt.astype(float)
        npgt = m.imresize(npgt, (self.img_size, self.img_size), 'nearest', mode='F')
        npgt = npgt.astype(int)

        assert (np.all(classes == np.unique(npgt)))

        return torch.from_numpy(npgt).long()


    def __getitem__(self, index):
        base_name = self.files[index]
        img_file = self.root_path + '/JPEGImages/' + base_name
        gt_file  = self.root_path + '/SegmentationClass/' + base_name

        img = self._get_image(img_file)

        realA = load_img(img_file)
        realA = self.transform(realA)
        if self.set=='train':
            if os.path.exists(gt_file):
                gt = self._get_gt(gt_file)
                realB = load_img(gt_file)
                realB = self.transform(realB)
                return img, gt,realA,realB,True
            else:
                return img,0,realA,0,False
        else:
            return img, realA, base_name


    def decode_segmap(self, temp, plot=False):
        label_colours = self._get_pascal_labels()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = (r/255.0)
        rgb[:, :, 1] = (g/255.0)
        rgb[:, :, 2] = (b/255.0)
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb


if __name__ == '__main__':
    VOC2012_PATH = './VOC2012'
    ds_train = VOC2012Dataset(VOC2012_PATH, set='train')
    loader_train = data_utils.DataLoader(ds_train,
                                         batch_size=1,
                                         num_workers=1, shuffle=True)
    k=0
    for i, (images, gts, realA, realB, status) in enumerate(loader_train):
        if status[0]==0:
            k+=1
    print(k)
### EOF ###
