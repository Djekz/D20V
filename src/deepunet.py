# deepunet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeepUNet, self).__init__()

        # Encoder
        self.enc_conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc_conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck_conv = DoubleConv(256, 512)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = DoubleConv(128, 64)

        # Output layer
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc_conv1(x)
        enc1_pool = self.pool1(enc1)
        enc2 = self.enc_conv2(enc1_pool)
        enc2_pool = self.pool2(enc2)
        enc3 = self.enc_conv3(enc2_pool)
        enc3_pool = self.pool3(enc3)

        # Bottleneck
        bottleneck = self.bottleneck_conv(enc3_pool)

        # Decoder
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec_conv3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec_conv2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec_conv1(dec1)

        # Output layer
        output = self.output_conv(dec1)
        return output

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
