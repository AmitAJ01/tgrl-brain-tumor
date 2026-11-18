"""
Baseline Brain Tumor Classifier (Component B).
Uses ResNet18 or EfficientNet-B0 with custom 3-class linear head.
Includes pure PyTorch ResNet18 fallback if torchvision is unavailable.
"""

import torch
import torch.nn as nn

try:
    import torchvision.models as models
    HAS_TORCHVISION = True
except ImportError:
    models = None
    HAS_TORCHVISION = False

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class PureResNet18(nn.Module):
    """Pure PyTorch implementation of ResNet-18 without external dependencies."""
    def __init__(self, in_channels: int = 3, num_classes: int = 1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def get_resnet18_backbone(pretrained: bool = True):
    """Instantiates ResNet18 using torchvision if present, or pure PyTorch fallback."""
    if HAS_TORCHVISION and models is not None:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        return models.resnet18(weights=weights)
    else:
        return PureResNet18()

class TumorClassifier(nn.Module):
    def __init__(self, backbone: str = "resnet18", num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet18":
            base = get_resnet18_backbone(pretrained=pretrained)
            in_features = base.fc.in_features if hasattr(base.fc, "in_features") else 512
            base.fc = nn.Identity()
            self.encoder = base
            self.classifier = nn.Linear(in_features, num_classes)
        elif backbone == "efficientnet_b0":
            if not HAS_TORCHVISION:
                raise ImportError("torchvision is required for efficientnet_b0")
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
