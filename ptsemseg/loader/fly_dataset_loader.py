DEBUG=True
def log(s):
    if DEBUG:
        print(s)

from torch.utils import data
from torchvision import transforms
import numpy as np
import torch
import collections
import os
from os.path import join as pjoin
from PIL import Image
from ptsemseg.utils import *
from scipy.ndimage.interpolation import zoom
from random import randint

class flyDatasetLoader(data.Dataset):
    def __init__(
        self,
        root,
        split,
        augmentations=None,
        data_split_info=None,
        patch_size=None
    ):
        self.root = os.path.expanduser(root)
        self.n_classes = 1 # for regression
        self.augmentations = augmentations
        self.data_split_info = data_split_info
        self.patch_size = [512, 512, 512] if patch_size is None else patch_size
        self.split = split

        # self.nameList = self.getInfoLists()
        self.pathList = self.getInfoLists()
        log('Init Data Loader: self.split is {}, length of self.pathList is {}'.format(self.split, len(self.pathList)))

    def __len__(self):
        return len(self.pathList)

    def __getitem__(self, index):
        img_dataset = self.pathList[index].split('/')[0]
        img_name = self.pathList[index].split('/')[1]
        img_path = pjoin(self.root, img_dataset, 'images', img_name)

        lbl_path = pjoin(self.root, img_dataset, 'labels', img_name)
        log('Loader: {}: img: {} label: {}'.format(index, img_path, lbl_path))
        img = loadtiff3d(img_path)
        lbl = loadtiff3d(lbl_path)
        log('before augmentation: img shape: {}'.format(img.shape))
        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)
        log('after augmentation: img shape: {}'.format(img.shape))
        # save augmentation
        # new_folder_maker(pjoin(self.root, img_dataset, 'augmentation'))
        # new_folder_maker(pjoin(self.root, img_dataset, 'augmentation_labels'))
        # writetiff3d(pjoin(self.root, img_dataset, 'augmentation', img_name + '_augmentation.tif'), img)
        # writetiff3d(pjoin(self.root, img_dataset, 'augmentation_labels', img_name + '_augmentation_label.tif'), lbl)
        ###
        img, lbl = self.find_patch(img, lbl)
        log('after find_patch: img shape: {}; should be 160x160x8'.format(img.shape))


        img, lbl = self.transform(img, lbl)

        return img, lbl


    def getInfoLists(self):

        if self.split == 'train':
            return [p for p in self.data_split_info['img_paths'] if p not in self.data_split_info['val_paths']]
        elif self.split == 'val':
            return self.data_split_info['val_paths']


    # find 160x160x8 patch for the image in given index
    def find_patch(self, img, lbl):

        shape = img.shape
        x = max(shape[0], self.patch_size[0]) / shape[0]
        y = max(shape[1], self.patch_size[1]) / shape[1]
        z = max(shape[2], self.patch_size[2]) / shape[2]
        img = zoom(img, (x, y, z))
        lbl = zoom(lbl, (x, y, z))
        x = randint(0, shape[0] - self.patch_size[0]) if x == 1 else 0
        y = randint(0, shape[1] - self.patch_size[1]) if y == 1 else 0
        z = randint(0, shape[2] - self.patch_size[2]) if z == 1 else 0

        img = img[x:x + self.patch_size[0], y:y + self.patch_size[1], z:z + self.patch_size[2]]
        lbl = lbl[x:x + self.patch_size[0], y:y + self.patch_size[1], z:z + self.patch_size[2]]

        return img, lbl

    # transform from numpy to tensor
    def transform(self, img, lbl):

        img = np.stack([img], axis=0)
        lbl = np.stack([lbl], axis=0)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()

        return img, lbl
