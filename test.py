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
# import random
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
# from random import randint
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

# transforms_tensor()
def tensor_array():
    a = np.ndarray([1,2,3])
    a = ToTensor()(a)
    print(a)
    a.view(1,1,2,3)
    print(a)
    # log('')

# tensor_array()

def usage_of_in():
    if 'a' in 'apple':
        print('yes')
# usage_of_in()

def usage_of_random_shuffle():
    from random import shuffle
    a = [1,2,3,4]
    b = a[:-1]
    shuffle(a)
    print(a)
    print(b)
    shuffle(b)
    print(b)
    print(a)

# usage_of_random_shuffle()

def usage_of_large_brackets():
    file_paths = 'file path'
    val_case_index = [1,2,3]
    case_index = [4,5,6]

    return {'file_paths': file_paths, 'val_case_index': val_case_index, 'case_index': case_index}

# a = usage_of_large_brackets()
# print(a)
# print(a['file_paths'])

def learn_yaml():
    import yaml
    with open('/Users/wonh/y3s2/isbi/segment_3D/configs/fcn3d_fly.yml') as fp:
        cfg = yaml.load(fp)
    # a = cfg['training']['patch_size']
    # a = a.split(',')
    # a = [int(tmp) for tmp in a]
    # print(a)
    # print(cfg['training'].get('patch_size'))
    # print(cfg['training'].get('patch_size', None))
    # print(cfg['training']['patch_size'].items())
    # patch_size = [para for axis, para in cfg['training']['patch_size'].items()]
    # print(patch_size)
    if cfg['model']['arch'] == 'fcn3dnet':
        print('yeah')

    # print(cfg.get('data'))
# learn_yaml()

 # def learn_seed():
 #     import random
 #     random.seed(1)
 #     print(random.randint(1, 10))
def learn_seed():
    import random
    random.seed(1)
    a = random.randint(1,10)
    print(a)
# learn_seed()

def learn_pjoin():
    root = '/Users/wonh/Desktop/flyJanelia'
    a = pjoin(root, 'images')
    print(a)
# learn_pjoin()

def learn_view():
    a = np.ndarray([3,3])
    # print(a.shape)
    a = ToTensor()(a)
    print(a)
    # a.view(1,3,3)
    # print(a)

# learn_view()

def learn_stack():
    a = np.ones(shape=(3,3,3))
    # print(a)
    b = np.ones(shape=(3, 3, 3))*99
    # a = np.stack([a], axis=3)
    a = np.stack([a,b], axis=0)
    print(a[1,:,:])
# learn_stack()

def learn_astype():
    lbl = np.zeros(shape=(3,3))
    lbl_background = (lbl == 0).astype('int')
    print(lbl_background)
    lbl = torch.from_numpy(lbl_background).long()
    print(lbl)
# learn_astype()

def test_label():
    lbl = loadtiff3d('/Users/wonh/Desktop/flyJanelia/labels/12.tif')
    lbl_background = (lbl == 0).astype('int')
    lbl_foreground = (lbl != 0).astype('int')
    lbl = np.stack([lbl_foreground, lbl_background], axis=3)
    print(lbl.shape)
    # print('lbl foreground sum: {} background sum: {}'.format(np.sum(lbl_background), np.sum(lbl_foreground)))
    # lbl = torch.from_numpy(lbl).long()

# test_label()

def test_matrix_add():
    a = np.asarray([1,2,3])
    a = a + 1
    print(a)
# test_matrix_add()

def test_zoom():
    a = np.ones(shape=(3,3,3))
    a[2][2][2] = 99
    print(a)
    a = torch.from_numpy(a).long()
    print(a.max())

# test_zoom()

def learn_makenp():
    from tensorboardX.summary import make_np
    scalar = [0.3]
    print(scalar)
    scalar = make_np(scalar)
    print(scalar.squeeze().ndim) # scalar.squeeze().ndim == 0

learn_makenp()