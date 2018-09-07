import torch.nn as nn
from ptsemseg.models.utils import *

class stem(nn.Module):
    def __init__(self, conv_in_channels):
        super(stem, self).__init__()

        self.layer1 = nn.Conv3d(in_channels=conv_in_channels, out_channels=32, kernel_size=(3,3,1), stride=2)
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
    def __init__(self, conv_in_channels):
        super(inceptionA, self).__init__()

        self.layer1 = nn.ReLU(inplace=False)
        self.layer2_1 = nn.Conv3d(in_channels=conv_in_channels, out_channels=192, kernel_size=(3,3,1), stride=2)
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
    def __init__(self, conv_in_channels):
        super(reductionA, self).__init__()

        self.layer1 = nn.ReLU(inplace=False)
        self.layer2_1 = nn.MaxPool3d(kernel_size=(3,3,1), stride=2)
        self.layer2_2 = nn.Conv3d(in_channels=conv_in_channels, out_channels=32, kernel_size=(3,3,1), stride=2)
        self.layer2_3 = nn.Sequential(nn.Conv3d(in_channels=conv_in_channels, out_channels=256, kernel_size=1, stride=1),
                                      nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3,3,1), stride=1),
                                      nn.Conv3d(in_channels=256, out_channels=384, kernel_size=(3,3,1), stride=1))



    def forward(self, x):
        out = self.layer1(x)
        out1 = self.layer2_1(out)
        out2 = self.layer2_2(out)
        out3 = self.layer2_3(out)
        out = torch.cat((out1, out2), dim=1)
        out = torch.cat((out, out3), dim=1)
        return x

class fcn3dnet(nn.Module):
    def __init__(self, num_classes=2):
        super(fcn3dnet, self).__init__()

        # stem
        self.block1 = stem(conv_in_channels=1)

        # inceptionA
        self.block2 = inceptionA(conv_in_channels=192)

        # reductionA
        self.block3 = reductionA(conv_in_channels=384)

        #todo

    def forward(self,x):
        print('The input size is: ' + str(x.size()))
        out = self.block1(x)
        print('The size after stem: ' + str(out.size))
        # out = self.block2(out)
        # print('The size after inceptionA: ' + str(out.size))
        # out = self.block3(out)
        # print('The size after reductionA: ' + str(out.size))

        return out

