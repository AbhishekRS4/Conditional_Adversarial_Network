import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet34

class ResnetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet34 = resnet34(pretrained=True)

    def forward(self, x):
        self.block1 = self.resnet34.conv1(x)
        self.block1 = self.resnet34.bn1(self.block1)
        self.block1 = self.resnet34.relu(self.block1)   # [64, H/2, W/2]

        self.block2 = self.resnet34.maxpool(self.block1)
        self.block3 = self.resnet34.layer1(self.block2)  # [64, H/4, W/4]
        self.block4 = self.resnet34.layer2(self.block3)  # [128, H/8, W/8]
        self.block5 = self.resnet34.layer3(self.block4)  # [256, H/16, W/16]
        self.block6 = self.resnet34.layer4(self.block6)  # [512, H/32, W/32]

class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        """
        TBD
        """
    def forward(self.):
        """
        TBD
        """

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        """
        TBD
        """
    def forward(self.):
        """
        TBD
        """

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        """
        TBD
        """
    def forward(self.):
        """
        TBD
        """

class ConditionalAdversarialNet(nn.Module):
    def __init__(self):
        """
        TBD
        """
    def forward(self.):
        """
        TBD
        """
