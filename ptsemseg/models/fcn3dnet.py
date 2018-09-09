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
                                   nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1,7,1), padding=(1,2,0)),
                                   nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(7,1,1), padding=(2,1,0)),
                                   nn.Conv3d(in_channels=64, out_channels=96, kernel_size=(3,3,1)))

    def forward(self, x):
        print('stem - input: ' + str(x.size()))
        conv1 = self.layer1(x)
        print('stem - after layer1: ' + str(conv1.size()))
        conv2 = self.layer2(conv1)
        print('stem - after layer2: ' + str(conv2.size()))
        conv3_1 = self.path1(conv2)
        print('stem - after layer2 path1: ' + str(conv3_1.size()))
        conv3_2 = self.path2(conv2)
        print('stem - after layer2 path2: ' + str(conv3_2.size()))
        out = torch.cat((conv3_1, conv3_2), dim=1) #not sure
        print('stem - after concate: ' + str(out.size()))
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
                                      nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3,3,1), padding=(1,1,0)),
                                      nn.Conv3d(in_channels=32, out_channels=384, kernel_size=1))
        self.layer3_3 = nn.Sequential(nn.Conv3d(in_channels=384, out_channels=32, kernel_size=1),
                                      nn.Conv3d(in_channels=32, out_channels=48, kernel_size=(3,3,1), padding=(1,1,0)),
                                      nn.Conv3d(in_channels=48, out_channels=64, kernel_size=(3,3,1), padding=(1,1,0)),
                                      nn.Conv3d(in_channels=64, out_channels=384, kernel_size=1))
    def forward(self, x):
        print('inceptionA - input: ' + str(x.size()))
        relu = self.layer1(x)
        print('inceptionA - after layer1: ' + str(relu.size()))
        conv1 = self.layer2_1(relu)
        print('inceptionA - after layer2_1: ' + str(conv1.size()))
        conv2 = self.layer2_2(relu)
        print('inceptionA - after layer2_2: ' + str(conv2.size()))
        concat = torch.cat((conv1, conv2), dim=1)
        print('inceptionA - after concat: ' + str(concat.size()))
        path3_1 = self.layer3_1(concat)
        print('inceptionA - after layer3_1: ' + str(path3_1.size()))
        path3_2 = self.layer3_2(concat)
        print('inceptionA - after layer3_2: ' + str(path3_2.size()))
        path3_3 = self.layer3_3(concat)
        print('inceptionA - after layer3_3: ' + str(path3_3.size()))
        out = torch.add(path3_1, path3_2)
        print('inceptionA - after concat first two: ' + str(out.size()))
        out = torch.add(out, path3_3)
        print('inceptionA - after concat the 3rd one: ' + str(out.size()))
        out = torch.add(out, concat)
        print('inceptionA - after concat the residual one: ' + str(out.size()))

        # residual- add input after relu to the final output

        return out

class reductionA(nn.Module):
    def __init__(self, conv_in_channels):
        super(reductionA, self).__init__()

        self.layer1 = nn.ReLU(inplace=False)
        self.layer2_1 = nn.MaxPool3d(kernel_size=(3,3,1), stride=2)
        self.layer2_2 = nn.Conv3d(in_channels=conv_in_channels, out_channels=128, kernel_size=(3,3,1), stride=2) # 32->128
        self.layer2_3 = nn.Sequential(nn.Conv3d(in_channels=conv_in_channels, out_channels=256, kernel_size=1, stride=1),
                                      nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3,3,1), stride=1),
                                      nn.Conv3d(in_channels=256, out_channels=384, kernel_size=(3,3,1), stride=2, padding=(1,1,0)))



    def forward(self, x):
        print('reductionA - input: ' + str(x.size()))
        out = self.layer1(x)
        print('reductionA - after layer1: ' + str(out.size()))
        out1 = self.layer2_1(out)
        print('reductionA - after layer2_1: ' + str(out1.size()))
        out2 = self.layer2_2(out)
        print('reductionA - after layer2_2: ' + str(out2.size()))
        out3 = self.layer2_3(out)
        print('reductionA - after layer2_3: ' + str(out3.size()))
        out = torch.cat((out1, out2), dim=1)
        print('reductionA - after concate path1 and path2: ' + str(out.size()))
        out = torch.cat((out, out3), dim=1)
        print('reductionA - after concate the 3rd path: ' + str(out.size()))
        return out

class inceptionB(nn.Module):
    def __init__(self, conv_in_channels):
        super(inceptionB, self).__init__()

        self.layer1 = nn.ReLU(inplace=False)
        self.layer2_1 = nn.Sequential(nn.Conv3d(in_channels=conv_in_channels, out_channels=128, kernel_size=1),
                                      nn.Conv3d(in_channels=128, out_channels=896, kernel_size=1))
        self.layer2_2 = nn.Sequential(nn.Conv3d(in_channels=conv_in_channels, out_channels=128, kernel_size=1),
                                      nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(1,7,1), padding=(1,2,0)),
                                      nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(7,1,1), padding=(2,1,0)),
                                      nn.Conv3d(in_channels=128, out_channels=896, kernel_size=1))
    def forward(self, x):
        print('inceptionB - input: ' + str(x.size()))
        out = self.layer1(x)
        print('inceptionB - after layer1: ' + str(out.size()))
        out1 = self.layer2_1(out)
        print('inceptionB - after layer2_1: ' + str(out1.size()))
        out2 = self.layer2_2(out)
        print('inceptionB - after layer2_2: ' + str(out2.size()))
        ret = torch.add(out1, out2)
        print('inceptionB - after adding 2_1 & 2_2: ' + str(out.size()))
        out = torch.add(ret, out)
        print('inceptionB - after adding residual path: ' + str(out.size()))

        # residual- add input after relu to the final output

        return out

class reductionB(nn.Module):
    def __init__(self, conv_in_channels):
        super(reductionB, self).__init__()

        self.layer1 = nn.ReLU(inplace=False)
        self.layer2_1 = nn.Sequential(nn.Conv3d(in_channels=conv_in_channels, out_channels=192, kernel_size=1),
                                      nn.Conv3d(in_channels=192, out_channels=192, kernel_size=(3,3,1), stride=2))
        self.layer2_2 = nn.MaxPool3d(kernel_size=(3,3,1), stride=2)
        self.layer2_3 = nn.Sequential(nn.Conv3d(in_channels=conv_in_channels, out_channels=256, kernel_size=1),
                                      nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(1,7,1), padding=(1,2,0)),
                                      nn.Conv3d(in_channels=256, out_channels=320, kernel_size=(7,1,1), padding=(2,1,0)),
                                      nn.Conv3d(in_channels=320, out_channels=960, kernel_size=(3,3,1), stride=2))

    def forward(self, x):
        print('reductionB - input: ' + str(x.size()))
        out = self.layer1(x)
        print('reductionB - after layer1: ' + str(out.size()))
        out1 = self.layer2_1(out)
        print('reductionB - after layer2_1: ' + str(out1.size()))
        out2 = self.layer2_2(out)
        print('reductionB - after layer2_2: ' + str(out2.size()))
        out3 = self.layer2_3(out)
        print('reductionB - after layer2_3: ' + str(out3.size()))
        out = torch.cat((out1, out2),dim=1)
        print('reductionB - after concatenating layer2_1 & layer2_2: ' + str(out.size()))
        out = torch.cat((out, out3), dim=1)
        print('reductionB - after concatenating  & layer2_3: ' + str(out.size()))

        return out


class inceptionC(nn.Module):
    def __init__(self, conv_in_channels):
        super(inceptionC, self).__init__()

        self.layer1 = nn.ReLU(inplace=False)
        self.layer2_1 = nn.Sequential(nn.Conv3d(in_channels=conv_in_channels, out_channels=192, kernel_size=1),
                                      nn.Conv3d(in_channels=192, out_channels=2048, kernel_size=1))
        self.layer2_2 = nn.Sequential(nn.Conv3d(in_channels=conv_in_channels, out_channels=192, kernel_size=1),
                                      nn.Conv3d(in_channels=192, out_channels=224, kernel_size=(1,3,1), padding=(0,1,0)),
                                      nn.Conv3d(in_channels=224, out_channels=256, kernel_size=(3,1,1), padding=(1,0,0)),
                                      nn.Conv3d(in_channels=256, out_channels=2048, kernel_size=1))

    def forward(self, x):
        print('inceptionC - input: ' + str(x.size()))
        out = self.layer1(x)
        print('inceptionC - after layer1: ' + str(out.size()))
        out1 = self.layer2_1(out)
        print('inceptionC - after layer2_1: ' + str(out1.size()))
        out2 = self.layer2_2(out)
        print('inceptionC - after layer2_2: ' + str(out2.size()))

        # out = torch.cat((out1, out2),dim=1)
        # print('inceptionC - after concatenating layer2_1 & layer2_2: ' + str(out.size()))

        ret = torch.add(out1, out2)
        print('inceptionC - after adding layer2_1 & layer2_2: ' + str(ret.size()))
        out = torch.add(ret, out) # residual- add input after relu to the final output
        print('inceptionC - after adding residual path: ' + str(out.size()))



        return out

class fcn3dnet(nn.Module):
    def __init__(self, num_classes=2):
        super(fcn3dnet, self).__init__()

        # stem
        self.block1 = stem(conv_in_channels=1)

        # inceptionA
        # self.block2 = inceptionA(conv_in_channels=192)
        self.block2 = inceptionA(conv_in_channels=192)

        # reductionA
        self.block3 = reductionA(conv_in_channels=384)

        # inceptionB
        self.block4 = inceptionB(conv_in_channels=896)

        # reductionB
        self.block5 = reductionB(conv_in_channels=896)

        # inceptionC
        self.block6 = inceptionC(conv_in_channels=2048)
        #todo

    def forward(self,x):
        print('The input size is: ' + str(x.size()))
        out = self.block1(x)
        print('The size after stem: ' + str(out.size()))
        out = self.block2(out)
        print('The size after inceptionA: ' + str(out.size()))
        out = self.block3(out)
        print('The size after reductionA: ' + str(out.size()))
        out = self.block4(out)
        print('The size after inceptionB: ' + str(out.size()))
        out = self.block5(out)
        print('The size after reductionB: ' + str(out.size()))
        out = self.block6(out)
        print('The size after inceptionC: ' + str(out.size()))

        return out


