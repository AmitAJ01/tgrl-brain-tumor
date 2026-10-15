"""
Tumor Segmentation Network (Component C).
ResNet18-Encoder U-Net architecture.
Produces tumor probability map M = S(X) in [0, 1]^(H x W).
"""

import torch
import torch.nn as nn
import torchvision.models as models

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        total_in = in_channels + skip_channels
        self.conv = nn.Sequential(
            nn.Conv2d(total_in, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            # Handle slight dimension mismatch if any
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResNetUNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Encoder stages
        self.init_conv = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        ) # [B, 64, 112, 112]
        self.maxpool = resnet.maxpool # [B, 64, 56, 56]

        self.layer1 = resnet.layer1 # [B, 64, 56, 56]
        self.layer2 = resnet.layer2 # [B, 128, 28, 28]
        self.layer3 = resnet.layer3 # [B, 256, 14, 14]
        self.layer4 = resnet.layer4 # [B, 512, 7, 7] - bottleneck

        # Decoder stages
        self.dec4 = DecoderBlock(512, 256, 256) # 7x7 -> 14x14
        self.dec3 = DecoderBlock(256, 128, 128) # 14x14 -> 28x28
        self.dec2 = DecoderBlock(128, 64, 64)   # 28x28 -> 56x56
        self.dec1 = DecoderBlock(64, 64, 32)    # 56x56 -> 112x112
        self.dec0 = DecoderBlock(32, 0, 16)     # 112x112 -> 224x224

        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = self.init_conv(x)     # 112x112, 64
        x_pool = self.maxpool(x0)  # 56x56, 64
        x1 = self.layer1(x_pool)   # 56x56, 64
        x2 = self.layer2(x1)       # 28x28, 128
        x3 = self.layer3(x2)       # 14x14, 256
        x4 = self.layer4(x3)       # 7x7, 512

        # Decoder
        d4 = self.dec4(x4, x3)     # 14x14, 256
        d3 = self.dec3(d4, x2)     # 28x28, 128
        d2 = self.dec2(d3, x1)     # 56x56, 64
        d1 = self.dec1(d2, x0)     # 112x112, 32
        d0 = self.dec0(d1, None)   # 224x224, 16

        logits = self.final_conv(d0)
        prob_map = self.sigmoid(logits)
        return prob_map
