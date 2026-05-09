
"""
04_cnn_model.py — CNN architecture for landscape image classification.
Uses a pretrained ResNet-18 backbone with partial fine-tuning:
  - Layers 1-3: FROZEN (low-level features transfer well)
  - Layer 4:    UNFROZEN (adapts high-level features to scatter plots)
  - Head:       TRAINABLE (classification layers)
Falls back to a lightweight custom CNN if torchvision is unavailable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CNN_CHANNELS, CNN_DENSE_DIMS, DROPOUT_RATE, N_CLASSES


try:
    import torchvision.models as models
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> MaxPool block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class LandscapeCNN(nn.Module):
    """Lightweight custom CNN fallback."""
    def __init__(self, n_classes=N_CLASSES, channels=CNN_CHANNELS,
                 dense_dims=CNN_DENSE_DIMS, dropout_rate=DROPOUT_RATE,
                 in_channels=3):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        prev_ch = in_channels
        for out_ch in channels:
            self.conv_blocks.append(ConvBlock(prev_ch, out_ch))
            prev_ch = out_ch

        self.last_channels = channels[-1]
        self.gap = nn.AdaptiveAvgPool2d(1)

        layers = []
        in_dim = self.last_channels
        for out_dim in dense_dims:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ]
            in_dim = out_dim

        self.dense = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return self.classifier(x)

    def get_penultimate_embedding(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x


class ResNetBackbone(nn.Module):
    """
    Pretrained ResNet-18 with partial fine-tuning.
    Layers 1-3 are frozen (low-level edge/texture features transfer well).
    Layer 4 is unfrozen (high-level features adapt to scatter plot domain).
    Classification head is fully trainable.
    """
    def __init__(self, n_classes=N_CLASSES, dropout_rate=DROPOUT_RATE):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze everything first
        for param in backbone.parameters():
            param.requires_grad = False

        # Unfreeze layer4 so it can adapt to scatter-plot features
        for param in backbone.layer4.parameters():
            param.requires_grad = True

        # Remove original FC layer
        feat_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Trainable classification head
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, n_classes),
        )

        # For Grad-CAM: expose the last conv layer
        self.target_layer = self.backbone.layer4[-1]

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def get_penultimate_embedding(self, x):
        features = self.backbone(x)
        # Return the 128-dim embedding before the final linear
        for layer in list(self.head.children())[:-1]:
            features = layer(features)
        return features


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(n_classes=N_CLASSES, in_channels=3):
    if _HAS_TORCHVISION:
        print("Using pretrained ResNet-18 (layer4 unfrozen) + trainable head")
        return ResNetBackbone(n_classes=n_classes)
    else:
        print("torchvision not available — using lightweight custom CNN")
        return LandscapeCNN(n_classes=n_classes, in_channels=in_channels)


if __name__ == "__main__":
    model = build_model()
    print("Model architecture:")
    print(model)
    print(f"\nTrainable params: {count_parameters(model):,}")

    batch = torch.randn(8, 3, 128, 128)
    logits = model(batch)
    print(f"Input: {batch.shape} -> Output: {logits.shape}")

    emb = model.get_penultimate_embedding(batch)
    print(f"Embedding: {emb.shape}")
