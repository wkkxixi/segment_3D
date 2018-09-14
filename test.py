import torch
#
# x = torch.randn(2, 3)
# print(torch.cat((x,x,x), 0).size())
# print(torch.cat((x,x,x), 1).size())
# def foo():
#     assert 0 == 1
#
# foo()

import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps

# zz = np.random.random((3,3))
# mask = zz.copy()
# mask = Image.fromarray(mask, mode="L")
# print(zz)
# img = Image.fromarray(zz, mode="RGB")
# print(img.size)
# print(np.array(img))
# print(np.array(mask, dtype=np.uint8))
# img_size = ('same', 'same')
# img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
# print(img_size)
# with open('/Users/wonh/Desktop/flyJanelia/datainfo/datainfo.txt') as f:
#     content = f.readlines()
# # you may also want to remove whitespace characters like `\n` at the end of each line
# content = [x.strip() for x in content]
# for c in content:
#     print(c.split()[0])

# im = Image.open('/Users/wonh/Desktop/flyJanelia/images/1.tif')
# print(np.array(im).shape)

from PIL import Image
from torchvision.transforms import ToTensor
from utils.io import *
import numpy as np
from scipy.ndimage.interpolation import zoom
from os.path import join as pjoin
from random import randint
from torchvision import transforms

# how to keep the image same after tensor
def np_tensor_np():
    image = loadtiff3d('/Users/wonh/Desktop/flyJanelia/images/1.tif') # x, y, z
    image = ToTensor()(image) # z, x, y
    image = image.numpy()*255
    image = image.astype('uint8') # important
    image = np.transpose(image, (1,2,0)) # z, x, y => x, y, z

def os_expand_user():
    root = '/Users/wonh/Desktop/flyJanelia/'
    path = os.path.expanduser(root)


def getInfoLists(root):
    nameList = []
    xList = []
    yList = []
    zList = []
    with open(root + '/datainfo/datainfo.txt') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for c in content:
        nameList.append(c.split()[0])
        xList.append(c.split()[1])
        yList.append(c.split()[2])
        zList.append(c.split()[3])
    return nameList, xList, yList, zList

# print(pjoin('/Users/wonh/Desktop/flyJanelia', 'images', '1.tif'))

def resize_small_img(filepath):
    img = loadtiff3d(filepath)
    x = max(img.shape[0], 160)/img.shape[0]
    y = max(img.shape[1], 160)/img.shape[1]
    z = max(img.shape[2], 8)/img.shape[2]
    print((x,y,z))
    img = zoom(img, (x,y,z))

    x = randint(0, img.shape[0]-160) if x == 1 else 0
    y = randint(0, img.shape[1]-160) if y == 1 else 0
    z = randint(0, img.shape[2] - 8) if z == 1 else 0
    print((x,y,z))
    img = img[x:x+160, y:y+160, z:z+8]
    print(img.shape)

    # img = np.resize(img, (x,y,z))
    writetiff3d('/Users/wonh/Desktop/test2.tif', img)

# resize_small_img('/Users/wonh/Desktop/flyJanelia/images/12.tif')

def transforms_tensor():
    tf = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])])
    img = loadtiff3d('/Users/wonh/Desktop/flyJanelia/labels/12.tif')

    img = tf(img)
    img = img.numpy() * 255
    img = img.astype('uint8')  # important
    img = np.transpose(img, (1, 2, 0))  # z, x, y => x, y, z
    writetiff3d('/Users/wonh/Desktop/test3.tif', img)
    print(img)

transforms_tensor()
