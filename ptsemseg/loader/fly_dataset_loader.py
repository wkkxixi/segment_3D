DEBUG=False
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
        img_path = self.root + '/' + img_dataset + '/' + 'images' + '/' + img_name
        lbl_path = self.root + '/' + img_dataset + '/' + 'labels' + '/' + img_name
        log('Loader: {}: img: {} label: {}'.format(index, img_path, lbl_path))
        img = loadtiff3d(img_path)
        lbl = loadtiff3d(lbl_path)
        print('whole => img max {}, min {} | label max {}, min {}'.format(np.max(img), np.min(img), np.max(lbl),
                                                                          np.min(lbl)))
        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        img, lbl = self.find_patch(img, lbl)

        # print('patch => img max {}, min {} | label max {}, min {}'.format(np.max(img), np.min(img), np.max(lbl), np.min(lbl)))

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
        zoomx = max(shape[0], self.patch_size[0]) / shape[0]
        zoomy = max(shape[1], self.patch_size[1]) / shape[1]
        zoomz = max(shape[2], self.patch_size[2]) / shape[2]
        # print('before zoom => img max {}, min {} | label max {}, min {}'.format(np.max(img), np.min(img), np.max(lbl),
        #                                                                   np.min(lbl)))
        img = zoom(img, (zoomx, zoomy, zoomz))
        lbl = zoom(lbl, (zoomx, zoomy, zoomz))
        # print('after zoom => img max {}, min {} | label max {}, min {}'.format(np.max(img), np.min(img), np.max(lbl),
        #                                                                   np.min(lbl)))
        x = randint(0, shape[0] - self.patch_size[0]) if zoomx >= 1 else 0
        y = randint(0, shape[1] - self.patch_size[1]) if zoomy >= 1 else 0
        z = randint(0, shape[2] - self.patch_size[2]) if zoomz >= 1 else 0
        # print('x: {} y: {} z: {}'.format(x, y, z))
        img_patch = img[x:x + self.patch_size[0], y:y + self.patch_size[1], z:z + self.patch_size[2]].copy()
        lbl_patch = lbl[x:x + self.patch_size[0], y:y + self.patch_size[1], z:z + self.patch_size[2]].copy()
        # print('lbl == 0 percentage: {}'.format(100*np.sum((lbl_patch == 0).astype('int'))/(self.patch_size[0]*self.patch_size[1]*self.patch_size[2])))
        # print('img max {}, min {} | label max {}, min {}'.format(np.max(img_patch), np.min(img_patch), np.max(lbl_patch),
        #                                                                   np.min(lbl_patch)))
        while np.sum((lbl_patch == 0).astype('int'))/(self.patch_size[0]*self.patch_size[1]*self.patch_size[2]) > 0.99995:
            x = randint(0, shape[0] - self.patch_size[0]) if zoomx >= 1 else 0
            y = randint(0, shape[1] - self.patch_size[1]) if zoomy >= 1 else 0
            z = randint(0, shape[2] - self.patch_size[2]) if zoomz >= 1 else 0
            # print('x: {} y: {} z: {} zoomx: {} zoomy: {} zoomz: {}'.format(x, y, z, zoomx, zoomy, zoomz))
            img_patch = img[x:x + self.patch_size[0], y:y + self.patch_size[1], z:z + self.patch_size[2]].copy()
            lbl_patch = lbl[x:x + self.patch_size[0], y:y + self.patch_size[1], z:z + self.patch_size[2]].copy()
            # print('sum img_patch: {}  sum lbl_patch: {}'.format(np.sum(img_patch), np.sum(lbl_patch)))
            # print('lbl == 0 percentage: {}'.format(100 * np.sum((lbl_patch == 0).astype('int')) / (
            #             self.patch_size[0] * self.patch_size[1] * self.patch_size[2])))
            # print('img max {}, min {} | label max {}, min {}'.format(np.max(img_patch), np.min(img_patch),
            #                                                          np.max(lbl_patch),
            #                                                          np.min(lbl_patch)))
        print('find nice patch!!!')

        # print('find_patch return  => img max {}, min {} | label max {}, min {}'.format(np.max(img), np.min(img), np.max(lbl),
        #                                                                   np.min(lbl)))
        # writetiff3d('/home/heng/Desktop/Research/isbi/fly-dataset/utokyofly/pred/1_img_patch.tif', img)
        # writetiff3d('/home/heng/Desktop/Research/isbi/fly-dataset/utokyofly/pred/1_lbl_patch.tif', lbl)
        return img_patch, lbl_patch/255

    # transform from numpy to tensor
    def transform(self, img, lbl):

        img = np.stack([img], axis=0)
        lbl = np.stack([lbl], axis=0)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()

        return img, lbl
