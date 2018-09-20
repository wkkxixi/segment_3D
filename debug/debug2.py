import numpy as np
import torch
from torch.autograd import Variable
from ptsemseg.models.inceptionv4 import *

inceptionv4_model = inceptionv4(num_classes=2)
# fcn3dnet_model.cuda()

fake_im_num = 20
numpy_fake_image = np.random.rand(fake_im_num, 3, 512, 512)
tensor_fake_image = torch.FloatTensor(numpy_fake_image)
print(tensor_fake_image.size())
torch_fake_image = Variable(tensor_fake_image)
output = inceptionv4_model(torch_fake_image)
