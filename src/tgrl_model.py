"""
Unified TGRL Architecture (Components D, E, F).
Shared Encoder F, U-Net Segmentation Decoder S, and Classification Head C.
Directly implements Figure 2 and Algorithm 1 of the paper.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from src.segmentation import DecoderBlock

class TGRLNetwork(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Shared Encoder F
        self.init_conv = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        ) # [B, 64, 112, 112]
        self.maxpool = resnet.maxpool # [B, 64, 56, 56]
        self.layer1 = resnet.layer1   # [B, 64, 56, 56]
        self.layer2 = resnet.layer2   # [B, 128, 28, 28]
        self.layer3 = resnet.layer3   # [B, 256, 14, 14]
        self.layer4 = resnet.layer4   # [B, 512, 7, 7] - bottleneck

        # Segmentation Decoder S
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        self.dec0 = DecoderBlock(32, 0, 16)
        self.seg_final = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Classification Head C (attached to bottleneck features)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def encode(self, x: torch.Tensor):
        x0 = self.init_conv(x)
        x_pool = self.maxpool(x0)
        x1 = self.layer1(x_pool)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4

    def decode_segmentation(self, x0, x1, x2, x3, x4):
        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)
        d0 = self.dec0(d1, None)
        logits = self.seg_final(d0)
        return self.sigmoid(logits)

    def classify_bottleneck(self, x4: torch.Tensor) -> torch.Tensor:
        feat = self.gap(x4).flatten(1)
        logits = self.classifier(feat)
        return logits

    def classify_image(self, x: torch.Tensor) -> torch.Tensor:
        """Runs image through shared encoder and classification head."""
        _, _, _, _, x4 = self.encode(x)
        return self.classify_bottleneck(x4)

    def forward(self, x: torch.Tensor, compute_roi: bool = False, compute_cf: bool = False):
        """
        Forward pass implementing Algorithm 1:
        1. Encodes x, produces mask M = S(x) and full logits p_full.
        2. If compute_roi: computes X_ROI = X ⊙ M, forwards through C(F(X_ROI)).
        3. If compute_cf: computes X_cf = X ⊙ (1 - M), forwards through C(F(X_cf)).
        """
        x0, x1, x2, x3, x4 = self.encode(x)
        mask = self.decode_segmentation(x0, x1, x2, x3, x4)
        logits_full = self.classify_bottleneck(x4)

        results = {
            "mask": mask,
            "logits_full": logits_full,
            "prob_full": torch.softmax(logits_full, dim=1)
        }

        if compute_roi:
            # Mask is [B, 1, H, W], broadcast to 3 channels: X_ROI = X ⊙ M
            x_roi = x * mask.detach()
            logits_roi = self.classify_image(x_roi)
            results["logits_roi"] = logits_roi
            results["prob_roi"] = torch.softmax(logits_roi, dim=1)
            results["x_roi"] = x_roi

        if compute_cf:
            # Counterfactual image: X_cf = X ⊙ (1 - M)
            x_cf = x * (1.0 - mask.detach())
            logits_cf = self.classify_image(x_cf)
            results["logits_cf"] = logits_cf
            results["prob_cf"] = torch.softmax(logits_cf, dim=1)
            results["x_cf"] = x_cf

        return results
