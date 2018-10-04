import numpy as np
import torch
from torch.autograd import Variable
from ptsemseg.models.fcn3dnet import *
from ptsemseg.models.unet3d import *
from ptsemseg.models.unet3dreg import *
from ptsemseg.models.unet3dregTeacher import *


def fcn3dnetDebug():

    fcn3dnet_model = fcn3dnet(num_classes=2)
    # fcn3dnet_model

    fake_im_num = 1
    numpy_fake_image = np.random.rand(fake_im_num, 1, 160, 160, 8)
    tensor_fake_image = torch.FloatTensor(numpy_fake_image)
    print(tensor_fake_image.size())
    torch_fake_image = Variable(tensor_fake_image)
    output = fcn3dnet_model(torch_fake_image)

def unet3dDebug():

    fcn3dnet_model = unet3d(n_classes=2)
    # fcn3dnet_model

    fake_im_num = 1
    numpy_fake_image = np.random.rand(fake_im_num, 1, 160, 160, 8)
    tensor_fake_image = torch.FloatTensor(numpy_fake_image)
    print(tensor_fake_image.size())
    torch_fake_image = Variable(tensor_fake_image)
    output = fcn3dnet_model(torch_fake_image)

def unet3dregDebug():
    fcn3dnet_model = unet3dreg(n_classes=1)
    # fcn3dnet_model

    fake_im_num = 1
    numpy_fake_image = np.random.rand(fake_im_num, 1, 160, 160, 8)
    tensor_fake_image = torch.FloatTensor(numpy_fake_image)
    print(tensor_fake_image.size())
    torch_fake_image = Variable(tensor_fake_image)
    output = fcn3dnet_model(torch_fake_image)

def unet3dregTeacherDebug():
    fcn3dnet_model = unet3dregTeacher(n_classes=1)
    # fcn3dnet_model

    fake_im_num = 1
    numpy_fake_image = np.random.rand(fake_im_num, 1, 160, 160, 8)
    tensor_fake_image = torch.FloatTensor(numpy_fake_image)
    print(tensor_fake_image.size())
    torch_fake_image = Variable(tensor_fake_image)
    output = fcn3dnet_model(torch_fake_image)

# unet3dregDebug()
unet3dregTeacherDebug()