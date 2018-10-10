DEBUG = False
def log(s):
    if DEBUG:
        print(s)
import torch.nn as nn

from ptsemseg.models.utils import *


class unet3dregSmartStudentRes(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        n_classes=1,
        is_deconv=True,
        in_channels=1,
        is_batchnorm=True
    ):
        super(unet3dregSmartStudentRes, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        # filters = [64, 128, 256, 512, 1024]
        filters = [64, 128, 256] # 16, 32, 64
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2_3d_regression(self.in_channels, filters[0], self.is_batchnorm)
        self.smartresConv1 = unetResConv_3d(filters[0], filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)
         # 1x1 convolutions are used to compute reductions before the expensive 3x3 convolutions
        self.conv_mid = nn.Conv3d(filters[0], filters[1], kernel_size=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)
        self.conv3 = unetConv2_3d_regression(filters[1], filters[2], self.is_batchnorm)
        self.smartresConv2 = unetResConv_3d(filters[2], filters[2], self.is_batchnorm)

        # upsampling
        self.up_concat2 = unetUp3d_regression_res(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp3d_regression_res(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.smartFinal = nn.Conv3d(filters[0], n_classes, 1)

        self.smartTanh = nn.Tanh()

    def forward(self, inputs):
        # log('unet3dregStudent: inputs size is {}'.format(inputs.size()))
        # conv1 = self.conv1(inputs)
        # log('unet3dregStudent: after conv1 size is {}'.format(conv1.size()))
        # maxpool1 = self.maxpool1(conv1)
        # log('unet3dregStudent: after maxpool1 size is {}'.format(maxpool1.size()))
        # conv_mid = self.conv_mid(maxpool1)
        # log('unet3dregStudent: after conv_mid size is {}'.format(conv_mid.size()))
        # maxpool2 = self.maxpool2(conv_mid)
        # log('unet3dregStudent: after maxpool2 size is {}'.format(maxpool2.size()))
        #
        # conv3 = self.conv3(maxpool2)
        # log('unet3dregStudent: after conv3 size is {}'.format(conv3.size()))
        #
        #
        #
        # up2 = self.up_concat2(conv_mid, conv3)
        # log('unet3dregStudent: after cat conv2 and conv_mid => up2 {}'.format(up2.size()))
        # up1 = self.up_concat1(conv1, up2)
        # log('unet3dregStudent: after cat conv1 and up2 => up1 {}'.format(up1.size()))
        #
        # final = self.smartFinal(up1)
        # log('unet3dregStudent: after final conv  => final {}'.format(final.size()))
        #
        # final = self.smartTanh(final)
        # log('unet3dregStudent: after tanh  => final {}'.format(final.size()))
        log('unet3dregStudent: inputs size is {}'.format(inputs.size()))
        conv1 = self.conv1(inputs)
        log('unet3dregStudent: after conv1 size is {}'.format(conv1.size()))
        resconv1 = self.smartresConv1(conv1)
        log('unet3dregStudent: after resconv1 size is {}'.format(resconv1.size()))
        maxpool1 = self.maxpool1(resconv1)
        log('unet3dregStudent: after maxpool1 size is {}'.format(maxpool1.size()))
        conv_mid = self.conv_mid(maxpool1)
        log('unet3dregStudent: after conv_mid size is {}'.format(conv_mid.size()))
        maxpool2 = self.maxpool2(conv_mid)
        log('unet3dregStudent: after maxpool2 size is {}'.format(maxpool2.size()))

        conv3 = self.conv3(maxpool2)
        log('unet3dregStudent: after conv3 size is {}'.format(conv3.size()))
        resconv2 = self.smartresConv2(conv3)
        log('unet3dregStudent: after resconv2 size is {}'.format(resconv2.size()))

        up2 = self.up_concat2(conv_mid, resconv2)
        log('unet3dregStudent: after cat conv2 and conv_mid => up2 {}'.format(up2.size()))
        up1 = self.up_concat1(conv1, up2)
        log('unet3dregStudent: after cat conv1 and up2 => up1 {}'.format(up1.size()))

        final = self.smartFinal(up1)
        log('unet3dregStudent: after final conv  => final {}'.format(final.size()))

        final = self.smartTanh(final)
        log('unet3dregStudent: after tanh  => final {}'.format(final.size()))

        return final, conv1, conv3, up2, up1
