import torch.nn as nn

from ptsemseg.models.utils import *


class stem(nn.Module):
    def __init__(self):
        self.layer1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3,3,1), stride=2)
        self.layer2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,3,4))

        self.path1 = nn.Sequential(nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
                                   nn.Conv3d(in_channels=64, out_channels=96, kernel_size=(3,3,1)))

        self.path2 = nn.Sequential(nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1),
                                   nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1,7,1)),
                                   nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1,7,1)),
                                   nn.Conv3d(in_channels=64, out_channels=96, kernel_size=(3,3,1)))

    def forward(self, x):
        conv1 = self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3_1 = self.path1(conv2)
        conv3_2 = self.path2(conv2)
        out = torch.cat((conv3_1, conv3_2), dim=1) #not sure
        return out


class inceptionA(nn.Module):
    def __init__(self):
        self.layer1 = nn.ReLU(inplace=False)
        self.layer2_1 = nn.Conv3d(in_channels=192, out_channels=192, kernel_size=(3,3,1), stride=2)
        self.layer2_2 = nn.MaxPool3d(kernel_size=(3,3,1), stride=2)
        self.layer3_1 = nn.Sequential(nn.Conv3d(in_channels=384, out_channels=32, kernel_size=1),
                                      nn.Conv3d(in_channels=32, out_channels=384, kernel_size=1))
        self.layer3_2 = nn.Sequential(nn.Conv3d(in_channels=384, out_channels=32, kernel_size=1),
                                      nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3,3,1)),
                                      nn.Conv3d(in_channels=32, out_channels=384, kernel_size=1))
        self.layer3_3 = nn.Sequential(nn.Conv3d(in_channels=384, out_channels=32, kernel_size=1),
                                      nn.Conv3d(in_channels=32, out_channels=48, kernel_size=(3,3,1)),
                                      nn.Conv3d(in_channels=48, out_channels=64, kernel_size=(3,3,1)),
                                      nn.Conv3d(in_channels=64, out_channels=384, kernel_size=1))
    def forward(self, x):
        relu = self.layer1(x)
        conv1 = self.layer2_1(relu)
        conv2 = self.layer2_2(relu)
        concat = torch.cat((conv1, conv2), dim=1)
        path3_1 = self.layer3_1(concat)
        path3_2 = self.layer3_2(concat)
        path3_3 = self.layer3_3(concat)
        out = torch.add(path3_1, path3_2)
        out = torch.add(out, path3_3)
        out = torch.add(out, concat)

        return out

class reductionA(nn.Module):
    def __init__(self):
        self.layer1 = nn.ReLU(inplace=False)
        self.layer2_1 = nn.MaxPool3d(kernel_size=(3,3,1), stride=2)
        # self.layer2_2 = nn.Conv3d(in_channels=, out_channels=, kernel_size=)

        


class fcn3dnet(nn.Module):
    def __init__(
            self,
            n_classes=2,
            is_deconv=True,
            in_channels=1,
            is_batchnorm=True,
    ):
        super(fcn3dnet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm

        filters = [64, 128, 256, 512, 1024]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final
