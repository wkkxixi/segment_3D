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

class flyJaneliaRegLoader(data.Dataset):
    def __init__(
        self,
        root,
        split,
        is_transform=False,
        # img_size=(480, 640),
        augmentations=None,
        img_norm=True,
        data_split_info=None,
        patch_size=None
    ):
        self.root = os.path.expanduser(root)
        self.is_transform = is_transform
        # self.n_classes = 2
        self.n_classes = 1 # for regression
        self.augmentations = augmentations
        self.data_split_info = data_split_info
        self.patch_size = 512 if patch_size is None else patch_size
        self.split = split
        # self.nameList, self.xList, self.yList, self.zList = self.getInfoLists()
        self.nameList = self.getInfoLists()

        # self.tf = transforms.Compose([transforms.ToTensor(),
        #                               transforms.Normalize([0.485, 0.456, 0.406],
        #                                                    [0.229, 0.224, 0.225])])
        self.tf = transforms.ToTensor()

        # self.shape = None


    def __len__(self):
        return len(self.nameList)

    def __getitem__(self, index):
        im_name = self.nameList[index]
        im_path = pjoin(self.root, 'images', im_name + '.tif')
        lbl_path = pjoin(self.root, 'labels_v3', im_name + '.tif')
        log('=========>  {}'.format(im_path))
        img = loadtiff3d(im_path)
        lbl = loadtiff3d(lbl_path)
        # self.shape = img.shape
        log('before augmentation: img shape: {}'.format(img.shape))
        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)
        log('after augmentation: img shape: {}'.format(img.shape))
        #writetiff3d(pjoin(self.root, 'augmentation', im_name + '_augmentation.tif'), img)
        #writetiff3d(pjoin(self.root, 'augmentation_labels', im_name + '_augmentation_label.tif'), lbl)
        img, lbl = self.find_patch(img, lbl)
        log('after find_patch: img shape: {}; should be 160x160x8'.format(img.shape))


        img, lbl = self.transform(img, lbl)

        # if self.is_transform:
        #     im, lbl = self.transform(im, lbl)
        return img, lbl

    # def getInfoLists(self):
    #     log('val_indices: {}'.format(self.data_split_info['val_indices']))
    #     nameList = []
    #     # xList = []
    #     # yList = []
    #     # zList = []
    #     with open(pjoin(self.root, 'datainfo', 'datainfo.txt')) as f:
    #         content = f.readlines()
    #     content = [x.strip() for x in content]
    #     for c in content:
    #         if self.split == 'train':
    #             log('loader init: train')
    #             if (c.split()[0]).split('.tif')[0] in self.data_split_info['val_indices']:
    #                 continue
    #         elif self.split == 'val':
    #             log('loader init: val')
    #             if not (c.split()[0]).split('.tif')[0] in self.data_split_info['val_indices']:
    #                 continue
    #         nameList.append((c.split()[0]).split('.tif')[0])
    #         # xList.append(int(c.split()[1]))
    #         # yList.append(int(c.split()[2]))
    #         # zList.append(int(c.split()[3]))
    #     log('loader init for {} has nameList({})'.format(self.split, len(nameList)))
    #     # return nameList, xList, yList, zList
    #     return nameList

    def getInfoLists(self):
        log('val_indices: {}'.format(self.data_split_info['val_indices']))
        nameList = []

        with open(pjoin(self.root, 'datainfo', 'datainfo.txt')) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        for c in content:
            if self.split == 'train':
                log('loader init: train')
                if (c.split()[0]).split('.tif')[0] in self.data_split_info['val_indices']:
                    continue
            elif self.split == 'val':
                log('loader init: val')
                if not (c.split()[0]).split('.tif')[0] in self.data_split_info['val_indices']:
                    continue
            nameList.append((c.split()[0]).split('.tif')[0])
        log('loader init for {} has nameList({})'.format(self.split, len(nameList)))
        return nameList

    # find 160x160x8 patch for the image in given index
    def find_patch(self, img, lbl):
        # x = max(self.xList[index], 160)/self.xList[index]
        # y = max(self.yList[index], 160)/self.yList[index]
        # z = max(self.zList[index], 8)/self.zList[index]
        # img = zoom(img, (x,y,z))
        # lbl = zoom(lbl, (x,y,z))
        # x = randint(0, img.shape[0] - 160) if x == 1 else 0
        # y = randint(0, img.shape[1] - 160) if y == 1 else 0
        # z = randint(0, img.shape[2] - 8) if z == 1 else 0
        #
        # img = img[x:x + 160, y:y + 160, z:z + 8]
        # lbl = lbl[x:x + 160, y:y + 160, z:z + 8]
        shape = img.shape
        x = max(shape[0], 160) / shape[0]
        y = max(shape[1], 160) / shape[1]
        z = max(shape[2], 8) / shape[2]
        img = zoom(img, (x, y, z))
        lbl = zoom(lbl, (x, y, z))
        x = randint(0, shape[0] - 160) if x == 1 else 0
        y = randint(0, shape[1] - 160) if y == 1 else 0
        z = randint(0, shape[2] - 8) if z == 1 else 0

        img = img[x:x + 160, y:y + 160, z:z + 8]
        lbl = lbl[x:x + 160, y:y + 160, z:z + 8]

        return img, lbl

    # transform from numpy to tensor
    def transform(self, img, lbl):
        # img = self.tf(img)
        # lbl = self.tf(lbl)

        img = np.stack([img], axis=0)
        # lbl = (lbl > 0).astype('int')
        lbl = np.stack([lbl], axis=0)
        img = torch.from_numpy(img).float()
        # lbl = torch.from_numpy(lbl).long()
        lbl = torch.from_numpy(lbl).float()

        return img, lbl

    # def transform(self, img, lbl):
    #     if self.img_size == ('same', 'same'):
    #         pass
    #     else:
    #         img = img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
    #         lbl = lbl.resize((self.img_size[0], self.img_size[1]))
    #     img = self.tf(img)
    #     lbl = torch.from_numpy(np.array(lbl)).long()
    #     lbl[lbl == 255] = 0
    #     return img, lbl