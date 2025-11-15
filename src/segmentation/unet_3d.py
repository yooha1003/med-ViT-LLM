"""
3D U-Net implementation for medical image segmentation
Based on the paper: "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block with batch normalization and ReLU activation"""

    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else None

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        if self.dropout:
            x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class DownBlock(nn.Module):
    """Downsampling block with max pooling"""

    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super(DownBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, dropout_p)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv_block(x)
        x = self.pool(skip)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block with transpose convolution"""

    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels, dropout_p)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        diff_d = skip.size()[2] - x.size()[2]
        diff_h = skip.size()[3] - x.size()[3]
        diff_w = skip.size()[4] - x.size()[4]

        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                     diff_h // 2, diff_h - diff_h // 2,
                     diff_d // 2, diff_d - diff_d // 2])

        x = torch.cat([skip, x], dim=1)
        x = self.conv_block(x)
        return x


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric medical image segmentation

    Args:
        in_channels (int): Number of input channels (e.g., 1 for single modality MRI)
        num_classes (int): Number of output segmentation classes
        base_features (int): Number of features in the first layer (default: 32)
        dropout_p (float): Dropout probability (default: 0.1)
    """

    def __init__(self, in_channels=1, num_classes=2, base_features=32, dropout_p=0.1):
        super(UNet3D, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder (downsampling path)
        self.down1 = DownBlock(in_channels, base_features, dropout_p)
        self.down2 = DownBlock(base_features, base_features * 2, dropout_p)
        self.down3 = DownBlock(base_features * 2, base_features * 4, dropout_p)
        self.down4 = DownBlock(base_features * 4, base_features * 8, dropout_p)

        # Bottleneck
        self.bottleneck = ConvBlock(base_features * 8, base_features * 16, dropout_p)

        # Decoder (upsampling path)
        self.up1 = UpBlock(base_features * 16, base_features * 8, dropout_p)
        self.up2 = UpBlock(base_features * 8, base_features * 4, dropout_p)
        self.up3 = UpBlock(base_features * 4, base_features * 2, dropout_p)
        self.up4 = UpBlock(base_features * 2, base_features, dropout_p)

        # Output layer
        self.out = nn.Conv3d(base_features, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        # Output
        x = self.out(x)

        return x

    def predict(self, x):
        """Generate segmentation prediction with softmax"""
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1)
        return prediction, probs
