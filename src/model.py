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
        self.block2 = self.resnet34.layer1(self.block2)  # [64, H/4, W/4]
        self.block3 = self.resnet34.layer2(self.block2)  # [128, H/8, W/8]
        self.block4 = self.resnet34.layer3(self.block3)  # [256, H/16, W/16]
        self.block5 = self.resnet34.layer4(self.block4)  # [512, H/32, W/32]
        return self.block5

class UNetDecoder(nn.Module):
    def __init__(self, encoder_net, out_channels=2):
        super().__init__()
        self.encoder_net = encoder_net
        self.up_block1 = self.up_conv_block(512, 256)
        self.conv_reduction_1 = nn.Conv2d(512, 256, kernel_size=1)

        self.up_block2 = self.up_conv_block(256, 128)
        self.conv_reduction_2 = nn.Conv2d(256, 128, kernel_size=1)

        self.up_block3 = self.up_conv_block(128, 64)
        self.conv_reduction_3 = nn.Conv2d(128, 64, kernel_size=1)

        self.up_block4 = self.up_conv_block(64, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        self.up_1 = self.up_block1(x)   # [256, H/16, W/16]
        self.up_1 = torch.cat([self.encoder_net.block4, self.up_1])     # [512, H/16, W/16]
        self.up_1 = self.conv_reduction_1(self.up_1)    # [256, H/16, W/16]

        self.up_2 = self.up_block2(self.up_2)   # [128, H/8, W/8]
        self.up_2 = torch.cat([self.encoder_net.block3, self.up_2])     # [256, H/8, H/8]
        self.up_2 = self.conv_reduction_2(self.up_2)    # [128, H/8, W/8]

        self.up_3 = self.up_block3(self.up_2)   # [64, H/4, W/4]
        self.up_3 = torch.cat([self.encoder_net.block2, self.up_3])     # [128, H/4, W/4]
        self.up_3 = self.conv_reduction_3(self.up_3)    # [64, H/4, W/4]

        self.up_4 = self.up_block4(self.up_3)   # [64, H/2, W/2]
        self.up_4 = torch.cat([self.encoder_net.block1, self.up_4])     # [128, H/2, W/2]
        self.up_4 = self.conv_reduction_3(self.up_4)    # [64, H/2, W/2]

        self.up_5 = self.up_block4(self.up_4)   # [64, H, W]
        self.out_features = self.final_conv(self.up_5)  # [2, H, W]
        return self.out_features

    def up_conv_block(self, in_channels, out_channels, conv_kernel_size=3, conv_tr_kernel_size=4):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=conv_tr_kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(),
            nn.ELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=conv_kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(),
            nn.ELU()
        )
        return block

class ResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_net = ResnetEncoder()
        self.decoder_net = UNetDecoder()

    def forward(self, x):
        self.encoder_features = self.encoder_net(x)
        self.decoder_features = self.decoder_net(self.encoder_features)
        return self.decoder_features

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.res_u_net = ResUNet()

    def forward(self, x):
        return self.res_u_net(x)

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
