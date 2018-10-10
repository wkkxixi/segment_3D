DEBUG = False
def log(s):
    if DEBUG:
        print(s)
import torch.nn as nn

from ptsemseg.models.utils import *


class unet3dregTeacher(nn.Module):
    def __init__(
        self,
        feature_scale=4,
        n_classes=1,
        is_deconv=True,
        in_channels=1,
        is_batchnorm=True,
    ):
        super(unet3dregTeacher, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        # filters = [64, 128, 256, 512, 1024]
        filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2_3d_regression(self.in_channels, filters[0], self.is_batchnorm)
        # residual between conv1 and maxpool1
        self.resConv1 = unetResConv_3d(filters[0], filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = unetConv2_3d_regression(filters[0], filters[1], self.is_batchnorm)
        # residual between conv2 and maxpool2
        self.resConv2 = unetResConv_3d(filters[1], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = unetConv2_3d_regression(filters[1], filters[2], self.is_batchnorm)
        # residual between conv3 and maxpool3
        self.resConv3 = unetResConv_3d(filters[2], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)

        # self.conv4 = unetConv2_3d(filters[2], filters[3], self.is_batchnorm)
        # self.maxpool4 = nn.MaxPool3d(kernel_size=2)

        # self.center = unetConv2_3d(filters[3], filters[4], self.is_batchnorm)
        self.center = unetConv2_3d_regression(filters[2], filters[3], self.is_batchnorm)
        # residual after center
        self.resConv4 = unetResConv_3d(filters[3], filters[3], self.is_batchnorm)

        # upsampling
        # self.up_concat4 = unetUp3d(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp3d_regression_res(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp3d_regression_res(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp3d_regression_res(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        self.tanh = nn.Tanh()

    def forward(self, inputs):

        log('unet3dregTeacher: inputs size is {}'.format(inputs.size()))
        conv1 = self.conv1(inputs)

        log('unet3dregTeacher: after conv1 size is {}'.format(conv1.size()))
        resconv1 = self.resConv1(conv1)
        log('unet3dregTeacher: after resConv1 size is {} should be the same'.format(resconv1.size()))
        maxpool1 = self.maxpool1(resconv1)
        log('unet3dregTeacher: after maxpool1 size is {}'.format(maxpool1.size()))

        conv2 = self.conv2(maxpool1)
        log('unet3dregTeacher: after conv2 size is {}'.format(conv2.size()))
        resconv2 = self.resConv2(conv2)
        log('unet3dregTeacher: after resConv2 size is {} should be the same'.format(resconv2.size()))
        maxpool2 = self.maxpool2(resconv2)
        log('unet3dregTeacher: after maxpool2 size is {}'.format(maxpool2.size()))

        conv3 = self.conv3(maxpool2)
        log('unet3dregTeacher: after conv3 size is {}'.format(conv3.size()))

        resconv3 = self.resConv3(conv3)
        log('unet3dregTeacher: after resConv3 size is {} should be the same'.format(resconv3.size()))
        maxpool3 = self.maxpool3(resconv3)
        log('unet3dregTeacher: after maxpool3 size is {}'.format(maxpool3.size()))

        # conv4 = self.conv4(maxpool3)
        # log('unet3d: after conv4 size is {}'.format(conv4.size()))
        # maxpool4 = self.maxpool4(conv4)
        # log('unet3d: after maxpool4 size is {}'.format(maxpool4.size()))

        center = self.center(maxpool3)
        log('unet3dregTeacher: after center => center {}'.format(center.size()))
        center = self.resConv4(center)
        log('unet3dregTeacher: after resConv4 size is {} should be the same'.format(center.size()))
        # up4 = self.up_concat4(conv3, center)
        # log('unet3d: after cat conv3 and center => up3 {}'.format(up4.size()))
        up3 = self.up_concat3(conv3, center)
        log('unet3dregTeacher: after cat conv3 and center => up3 {}'.format(up3.size()))
        up2 = self.up_concat2(conv2, up3)
        log('unet3dregTeacher: after cat conv2 and up3 => up2 {}'.format(up2.size()))

        up1 = self.up_concat1(conv1, up2)
        log('unet3dregTeacher: after cat conv1 and up2 => up1 {}'.format(up1.size()))


        final = self.final(up1)
        log('unet3dregTeacher: after final conv  => final {}'.format(final.size()))

        final = self.tanh(final)
        log('unet3dregTeacher: after tanh  => final {}'.format(final.size()))
        return final, conv1, conv3, up2, up1
