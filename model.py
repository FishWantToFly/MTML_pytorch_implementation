import torch
import torch.nn as nn
from torch.nn import init
import sys
import numpy as np
import torch.nn.functional as F


class MTML(nn.Module):
    def __init__(self):
        super(MTML, self).__init__()

        self.conv1 = nn.Conv3d(3, 8, kernel_size=7, padding = 3)
        self.act = nn.ReLU()
        self.conv2_1 = nn.Conv3d(8, 16, kernel_size=1)
        self.conv2_2 = nn.Conv3d(8, 16, kernel_size=3, padding = 1)
        self.conv3 = nn.Conv3d(16, 16, kernel_size=3, padding = 1)
        self.maxpool = nn.MaxPool3d(2)
        self.conv4 = nn.Conv3d(16, 32, kernel_size=3, padding = 1)
        self.conv5_1 = nn.Conv3d(32, 32, kernel_size=1)
        self.conv5_2 = nn.Conv3d(32, 32, kernel_size=3, padding = 1)
        self.conv6 = nn.Conv3d(32, 32, kernel_size=3, padding = 1)
        self.conv7 = nn.Conv3d(32, 32, kernel_size=3, padding = 1)
        self.conv8 = nn.Conv3d(32, 32, kernel_size=3, dilation=2, padding =2) # dilated conv.
        self.conv9 = nn.Conv3d(32, 32, kernel_size=3, dilation=2, padding =2)
        self.conv10 = nn.Conv3d(32, 32, kernel_size=3, dilation=2, padding =2)
        self.conv11 = nn.Conv3d(32, 32, kernel_size=3, dilation=2, padding =2)
        self.deconv1 = nn.ConvTranspose3d(96, 128, kernel_size=4, stride= 2 , padding = 1)
        self.conv12 = nn.Conv3d(128, 64, kernel_size=1)
        self.conv13_1 = nn.Conv3d(64, 3, kernel_size=1)
        self.conv13_2 = nn.Conv3d(64, 3, kernel_size=1)

    def forward(self, x):
        x = x.permute(0,4,1,2,3)

        out = self.conv1(x)
        out = self.act(out)
        out1 = self.conv2_1(out)
        out1 = self.act(out1)
        out = self.conv2_2(out)
        out = self.act(out)
        out = self.conv3(out)
        out = self.act(out)
        out = out1 + out 

        out = self.maxpool(out)
        out = self.conv4(out)
        out = self.act(out)
        out1 = self.conv5_1(out) 
        out1 = self.act(out1) 
        out = self.conv5_2(out)
        out = self.act(out) 
        out = out1 + out 
        out2 = out

        out = self.conv6(out)
        out = self.act(out)
        out = self.conv7(out)
        out = self.act(out)
        out = self.conv8(out)
        out = self.act(out)
        out = self.conv9(out)
        out = self.act(out)
        out3 = out

        out = self.conv10(out)
        out = self.act(out)
        out = self.conv11(out)
        out = self.act(out)

        out = torch.cat((out, out2, out3), dim =1)
        out = self.deconv1(out)
        out = self.conv12(out)
        out = self.act(out)

        dir_embedding = self.conv13_1(out)
        feature_embedding = self.conv13_2(out)

        dir_embedding = dir_embedding.permute(0, 2, 3, 4, 1)
        feature_embedding = feature_embedding.permute(0, 2, 3, 4, 1)
        return dir_embedding, feature_embedding