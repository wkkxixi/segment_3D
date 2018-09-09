import numpy as np
import torch
from torch.autograd import Variable
from ptsemseg.models.fcn3dnet import *

fcn3dnet_model = fcn3dnet(num_classes=2)
# fcn3dnet_model

fake_im_num = 20
numpy_fake_image = np.random.rand(fake_im_num, 1, 160, 160, 8)
tensor_fake_image = torch.FloatTensor(numpy_fake_image)
print(tensor_fake_image.size())
torch_fake_image = Variable(tensor_fake_image)
output = fcn3dnet_model(torch_fake_image)
