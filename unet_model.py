import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchsummary import summary
from torchvision import transforms
import os
from zipfile import ZipFile
import matplotlib.pyplot as plt
import numpy as np
from torcheval.metrics.functional import peak_signal_noise_ratio

torch.manual_seed(66)


class SkipConfig(nn.Module):
    def __init__(self, use_skip: bool = True) -> None:
        super(SkipConfig, self).__init__()
        self.use_skip = use_skip

    def forward(self, x, skip):
        if not self.use_skip:
            return x
        else:
            return torch.cat([x, skip], dim=1)


class FirstFeature(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FirstFeature, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)


class BaseBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size: tuple = (3, 3)) -> None:
        super(BaseBlock, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=self.kernel_size,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 base_block=BaseBlock):
        super(ConvBlock, self).__init__()
        self.base_block = base_block
        self.conv = nn.Sequential(
            self.base_block(in_channels,
                            out_channels,
                            kernel_size=(3, 3)),
            self.base_block(out_channels,
                            out_channels,
                            kernel_size=(3, 3))
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2) -> None:
        super(UpsamplingBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=scale_factor),
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=(1, 1),
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 skip_config=SkipConfig(use_skip=True)):
        super(Decoder, self).__init__()
        self.skip_config = skip_config
        skip_channels = out_channels
        if not skip_config.use_skip:
            skip_channels = out_channels * 2
        self.conv = UpsamplingBlock(in_channels, skip_channels, scale_factor=2)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.conv(x)
        x = self.skip_config(x, skip)
        x = self.conv_block(x)
        return x


class FinalOutput(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalOutput, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(
            self,
            n_channels=3,
            n_classes=3,
            skip_config=SkipConfig(use_skip=True),
    ):
        super(Unet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.in_conv1 = FirstFeature(n_channels, 64)
        self.in_conv2 = ConvBlock(64, 64)

        self.enc_1 = Encoder(64, 128)
        self.enc_2 = Encoder(128, 256)
        self.enc_3 = Encoder(256, 512)
        self.enc_4 = Encoder(512, 1024)

        self.dec_1 = Decoder(1024, 512, skip_config=skip_config)
        self.dec_2 = Decoder(512, 256, skip_config=skip_config)
        self.dec_3 = Decoder(256, 128, skip_config=skip_config)
        self.dec_4 = Decoder(128, 64, skip_config=skip_config)

        self.out_conv = FinalOutput(64, n_classes)

    def forward(self, x):
        x = self.in_conv1(x)
        x1 = self.in_conv2(x)

        x2 = self.enc_1(x1)
        x3 = self.enc_2(x2)
        x4 = self.enc_3(x3)
        x5 = self.enc_4(x4)

        x = self.dec_1(x5, x4)
        x = self.dec_2(x, x3)
        x = self.dec_3(x, x2)
        x = self.dec_4(x, x1)

        x = self.out_conv(x)

        return x


class SR_Unet(nn.Module):
    LOW_IMG_HEIGHT = 64
    LOW_IMG_WIDTH = 64

    def __init__(self,
                 n_channels=3,
                 n_classes=3,
                 skip_config=SkipConfig(use_skip=True)) -> None:
        super(SR_Unet, self).__init__()
        resize_shape = (self.LOW_IMG_HEIGHT * 4, self.LOW_IMG_WIDTH * 4)
        self.resize_fn = transforms.Resize(resize_shape, antialias=True)
        self.unet = Unet(
            n_channels=n_channels,
            n_classes=n_classes,
            skip_config=skip_config)

    def forward(self, x):
        x = self.resize_fn(x)
        x = self.unet(x)

        return x


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # SR Unet without skip connection
    unet_model = SR_Unet(skip_config=SkipConfig(use_skip=False)).to(device)
    img = torch.ones(2, 3, 64, 64).to(device)
    print(unet_model(img).shape)

    # SR Unet with skip connection
    unet_model = SR_Unet(skip_config=SkipConfig(use_skip=True)).to(device)
    img = torch.ones(2, 3, 64, 64).to(device)
    print(unet_model(img).shape)
