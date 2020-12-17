"""
denseNet implementation
"""
"""
FC-DenseNet44 model, growth rate k=12
"""

# Architecture: where m is channel size, k is growth rate on each layer, DB is dense block, TD is transition down, TU is transition up
"""
Input, m = 1
3 * 3 convolution, m = 16
DB(3 layers) + TD, m = 16 + 3 * k = 52
DB(4 layers) + TD, m = 16 + 7 * k = 100
DB(6 layers) + TD, m = 16 + 13 * k = 172
DB(9 layers), m = 16 + 21 * k = 268
TU + DB(6 layers), m = 9 * k + 172 + 6 * k = 340
TU + DB(4 layers), m = 6 * k + 100 + 4 * k = 220
TU + DB(3 layers), m = 4 * k + 52 + 3 * k = 136
1 * 1 convolution, m = c = 4
softmax, c = 4
"""

import torch
import torch.nn as nn


class denseLayer(nn.Module):
    def __init__(self, input_channel, k=12):
        super(denseLayer, self).__init__()

        self.bn = nn.BatchNorm2d(input_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_channel,
                              k,
                              3,
                              stride=1,
                              padding=1,
                              bias=True)
        self.dropOut = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.dropOut(x)

        return x


class denseBlock(nn.Module):
    def __init__(self, input_channel, layerNum, k=12, upsample=False):
        super(denseBlock, self).__init__()

        self.upsample = upsample
        self.denseLayers = nn.ModuleList(
            [denseLayer(input_channel + i * k, k) for i in range(layerNum)])

    def forward(self, x):
        output = []
        out = None

        if self.upsample:
            for layer in self.denseLayers:
                out = layer(x)
                x = torch.cat((out, x), dim=1)
                output.append(out)
        else:
            for layer in self.denseLayers:
                out = layer(x)
                x = torch.cat((out, x), dim=1)

        return x


class transitionDown(nn.Module):
    def __init__(self, input_channel):
        super(transitionDown, self).__init__()

        self.bn = nn.BatchNorm2d(input_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(input_channel, input_channel, 1)
        self.dropOut = nn.Dropout2d(0.2)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.dropOut(x)
        x = self.maxpool(x)

        return x


class transitionUp(nn.Module):
    def __init__(self, input_channel):
        super(transitionUp, self).__init__()

        self.conv = nn.ConvTranspose2d(input_channel, input_channel, (3, 3), 2)

    def forward(self, x):
        x = self.conv(x)
        return x


class FC_denseNet44(nn.Module):  # Define your model
    def __init__(self, k=12):
        super(FC_denseNet44, self).__init__()
        # fill in the constructor for your model here

        self.k = k  # growth rate for each layer as hyperparameter

        self.inpConv = nn.Conv2d(1, 16, (3, 3))
        self.outConv = nn.Conv2d(136, 4, (1, 1))
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        x = self.inpConv(x)
        """
        Part 1: Downsample path
        """
        input = x
        x = denseBlock(16, 3, self.k)
        concat_L1 = torch.cat((x, input), dim=1)
        layerNum_L1 = 16 + 3 * self.k
        x = transitionDown(16 + 3 * self.k)

        input = x
        x = denseBlock(16 + 3 * self.k, 4, self.k)
        concat_L2 = torch.cat((x, input), dim=1)
        layerNum_L2 = 16 + 7 * self.k
        x = transitionDown(16 + 7 * self.k)

        input = x
        x = denseBlock(16 + 7 * self.k, 6, self.k)
        concat_L3 = torch.cat((x, input), dim=1)
        layerNum_L3 = 16 + 13 * self.k
        x = transitionDown(16 + 13 * self.k)
        """
        Part 2: Bottle Neck
        """
        x = denseBlock(16 + 13 * self.k, 9, self.k)
        """
        Part 3: Upsample path
        """
        x = transitionUp(16 + 21 * self.k)
        x = torch.cat((concat_L3, x), dim=1)
        x = denseBlock(layerNum_L3 + 9 * self.k, 6, self.k, True)

        x = transitionUp(16 + 27 * self.k)
        x = torch.cat((concat_L2, x), dim=1)
        x = denseBlock(layerNum_L2 + 6 * self.k, 4, self.k, True)

        x = transitionUp(16 + 27 * self.k)
        x = torch.cat((concat_L1, x), dim=1)
        x = denseBlock(layerNum_L1 + 4 * self.k, 3, self.k, True)

        x = self.outConv(x)
        x = self.softmax(x)