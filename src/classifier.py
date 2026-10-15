"""
Baseline Brain Tumor Classifier (Component B).
Uses pretrained ResNet18 or EfficientNet-B0 with custom 3-class linear head.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class TumorClassifier(nn.Module):
    def __init__(self, backbone: str = "resnet18", num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.encoder = base
            self.classifier = nn.Linear(in_features, num_classes)
        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.encoder = base
            self.classifier = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        logits = self.classifier(feat)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
